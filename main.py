from math import log10, sqrt
import cv2
import numpy as np

#Calculate PSNR value
def PSNR(original, converted):
    mse = np.mean((original - converted) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

# Import picture & create RGB,HSI and YCbCr copies using algorithm
img = cv2.imread('mandrill.ppm', 1)
hsi =  cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)

cv2.imwrite('rgb.png', img)
cv2.imwrite('hsi.png', hsi)
cv2.imwrite('ycrcb.png', ycrcb)

# Display Converted Images
cv2.imshow('RGB Image',img)
cv2.imshow('HSI Image',hsi)
cv2.imshow('YCrCb Image',ycrcb)

# The three value channels
"""
cv2.imshow('R Channel', img[:, :, 0])
cv2.imshow('G Channel', img[:, :, 1])
cv2.imshow('B Channel', img[:, :, 2])

cv2.imshow('H Channel', hsi[:, :, 0])
cv2.imshow('S Channel', hsi[:, :, 1])
cv2.imshow('I Channel', hsi[:, :, 2])

cv2.imshow('Y Channel', ycrcb[:, :, 0])
cv2.imshow('Cr Channel', ycrcb[:, :, 1])
cv2.imshow('Cb Channel', ycrcb[:, :, 2])
"""
# Calculate PSNR Values for HSI and YCbCr with RGB

HSIvalue = PSNR(img, hsi)
print(f"PSNR value between RGB and HSI is {HSIvalue} dB")

YCrCbvalue = PSNR(img, ycrcb)
print(f"PSNR value between RGB and YrbCb is {YCrCbvalue} dB")


# Wait for a key press and then terminate the program
cv2.waitKey(0)
cv2.destroyAllWindows()