if mask_land[0,0]==1 or mask_land[0,224]==1:
    if roll > 0:
        roll = -180 + roll
    else:
        roll = 180 + roll