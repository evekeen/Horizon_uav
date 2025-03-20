import tensorflow as tf
import cv2
import numpy as np
import os
import math
import matplotlib.pyplot as plt
import torch
import requests
from pathlib import Path

# Enable GPU Memory Growth
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

def get_border(mask_land, mask_sky):
    """Get horizon border image from land and sky mask"""
    # Convert Colorspace to Grayscale
    mask_land = mask_land[:,:]
    mask_sky = mask_sky[:,:]
    # Get Horizon Border Using Dilation and Bitwise AND
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
    land_dilated = cv2.dilate(mask_land, kernel)
    sky_dilated = cv2.dilate(mask_sky, kernel)
    border = cv2.bitwise_and(land_dilated, sky_dilated)

    return border

def get_horizon_line(border):
    """Get horizon line equation from border image"""
    # Get border data in x,y format
    y = np.argmax(border, axis=0)
    x = np.arange(len(y))
    border_data = np.vstack([x, y]).T

    # Remove 0 from border data
    border_data = border_data[border_data[:, -1] != 0]

    # Linear Regression using border data
    # y = m*x+c
    x = border_data[:,0]
    y = border_data[:,1]

    X = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(X, y, rcond=None)[0]

    plt.figure(figsize=(5, 5))
    plt.scatter(x, y, label='Border Data')
    x = np.linspace(0, 224, 1000)
    y = m*x+c
    #plt.plot(x, y, '-r', label='Regression Line (y=m*x+c)')
    plt.legend(loc='upper left')
    plt.xlim(0, 224)
    plt.ylim(224, 0)
    plt.show()
    
    return m, c

def get_roll_pitch(m, c, image_height, image_width):
    """Get roll and pitch from horizon line equation"""
    # Convert slope (m) to roll degrees
    roll = math.degrees(math.atan(m))

    # Get pitch
    pitch = ((m*(image_width/2)+c)-(image_width/2))/(image_width/2)*100
    
    return roll, pitch

def draw_horizon_line(img, m, c, scale):
    """Draw horizon line on image"""
    image_height = img.shape[0]
    image_width = img.shape[1]

    c = scale*c

    pt1 = (0, int(m*0+c))
    pt2 = (image_width, int(m*image_width+c))

    cv2.line(img, pt1, pt2, (125, 0, 255), 2)

    return img

# Metric Function
class MaxMeanIoU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1), sample_weight)

# Loss Function
def dice_loss(y_true, y_pred, num_classes=2):
    smooth=tf.keras.backend.epsilon()
    dice=0
    for index in range(num_classes):
        y_true_f = tf.keras.backend.flatten(y_true[:,:,:,index])
        y_pred_f = tf.keras.backend.flatten(y_pred[:,:,:,index])
        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f)
        dice += (intersection + smooth) / (union + smooth)
    return -2./num_classes * dice

# Parameter
image_size = (224, 224)
model_path = os.path.join("model", "model-unet.h5")
image_path = os.path.join("golf4.jpg")
yolo_model_path = os.path.join("model", "yolov8n-seg.pt")

# Download YOLOv8 model if it doesn't exist
def download_yolo_model(model_path):
    if not os.path.exists(model_path):
        print(f"Downloading YOLOv8 model to {model_path}...")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt"
        with open(model_path, 'wb') as f:
            response = requests.get(url, stream=True)
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")

# Function to detect people in image and create a mask
def detect_people(image, yolo_model):
    """Detect people in the image and return a mask where people are present"""
    # Make a prediction with the model
    results = yolo_model(image)
    
    # Create an empty mask with the same size as the input image
    people_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    # Process results
    for result in results:
        # Check if there are any detections
        if len(result.boxes) > 0:
            # Get all person class detections (class 0 in COCO dataset is person)
            person_indices = [i for i, cls in enumerate(result.boxes.cls.cpu().numpy()) if cls == 0]
            
            # If any people were detected
            if person_indices and hasattr(result, 'masks') and result.masks is not None:
                # Get masks for people
                for idx in person_indices:
                    if idx < len(result.masks):
                        # Get the mask for this person
                        mask = result.masks[idx].data.cpu().numpy()[0]
                        mask = (mask > 0).astype(np.uint8) * 255
                        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
                        people_mask = cv2.bitwise_or(people_mask, mask)
    
    return people_mask

# Load horizon detection model
model = tf.keras.models.load_model(model_path, custom_objects={'dice_loss': dice_loss, 'MaxMeanIoU': MaxMeanIoU})

# Load and prepare YOLOv8 model
download_yolo_model(yolo_model_path)
try:
    yolo_model = torch.hub.load('ultralytics/yolov8', 'custom', path=yolo_model_path, verbose=False)
except:
    # Alternative method if torch.hub doesn't work
    from ultralytics import YOLO
    yolo_model = YOLO(yolo_model_path)

# Load image
frame = cv2.imread(image_path)
image_height = frame.shape[0]
image_width = frame.shape[1]
frame_ori = frame.copy()

# Detect people before resizing for horizon detection
people_mask = detect_people(frame_ori, yolo_model)

# Display people mask
cv2.imshow("People Detected", people_mask)

# Resize and prepare image for horizon detection
frame = cv2.resize(frame, image_size)
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
frame = cv2.normalize(frame, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)

# Predict mask
pred = model.predict(np.expand_dims(frame, 0))

# Process mask
mask = pred.squeeze()
mask = np.stack((mask,)*3, axis=-1)
mask[mask >= 0.5] = 1
mask[mask < 0.5] = 0

mask_land = mask[:, :, 0]
mask_sky = mask[:, :, 1]

# Resize people mask to match the processed image size
people_mask_resized = cv2.resize(people_mask, image_size)
people_mask_resized = people_mask_resized > 0

# Expand the people mask using dilation to create a buffer zone
dilation_kernel_size = 15  # Adjust this value to control how much to expand the mask
dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_kernel_size, dilation_kernel_size))
people_mask_expanded = cv2.dilate(people_mask_resized.astype(np.uint8), dilation_kernel, iterations=1)

# Create a copy of the expanded mask for visualization
people_mask_expanded_viz = people_mask_expanded.copy() * 255
cv2.imshow("Expanded People Mask", people_mask_expanded_viz)

# Remove people from land and sky masks using the expanded mask
mask_land[people_mask_expanded > 0] = 0
mask_sky[people_mask_expanded > 0] = 0

# Post Process
mask_land = cv2.cvtColor(mask_land, cv2.COLOR_BGR2GRAY)
mask_sky = cv2.cvtColor(mask_sky, cv2.COLOR_BGR2GRAY)

border = get_border(mask_land, mask_sky)
m, c = get_horizon_line(border)

resized_image_height = frame.shape[0]
resized_image_width = frame.shape[1]
print(resized_image_height, resized_image_width)
roll, pitch = get_roll_pitch(m, c, resized_image_height, resized_image_width)

if mask_land[0,0]==1 or mask_land[0,223]==1:
    if roll > 0:
        roll = -180 + roll
    else:
        roll = 180 + roll

#frame_ori = cv2.resize(frame_ori, (480, 480))
scale = image_height/image_size[0]
cv2.imshow("Original", frame_ori)

frame_ori = draw_horizon_line(frame_ori, m, c, scale)

text_roll = "roll:" + str(round(roll, 2)) + " degree"
text_pitch = "pitch:" + str(round(pitch, 2)) + " %"

cv2.putText(frame_ori, text_roll, (5, 15), 0, 0.5, (125, 0, 255), 2)
cv2.putText(frame_ori, text_pitch, (5, 35), 0, 0.5, (125, 0, 255), 2)

cv2.imshow("Horizon", frame_ori)
cv2.imshow("Land", mask_land)
cv2.imshow("Border", border)

cv2.waitKey(0)