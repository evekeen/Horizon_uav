import serial

ser = serial.Serial('/dev/ttyS0', 57600)

def send_attitude(roll=0, pitch=0, heading=0):
    data = [True, round(roll, 2), round(pitch, 2), round(heading, 2)]

    string = '#'
    for x in data:
        string += str(x)
        string += ","
    string += "&"
    ser.write(string.encode())