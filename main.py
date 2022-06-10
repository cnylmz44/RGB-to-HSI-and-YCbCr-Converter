from math import log10, sqrt
import cv2
import numpy as np

def rgb2hsi(rgb_lwpImg):
    rows = int(rgb_lwpImg.shape[0])
    cols = int(rgb_lwpImg.shape[1])
    b, g, r = cv2.split(rgb_lwpImg)
    b = b / 255.0
    g = g / 255.0
    r = r / 255.0
    hsi_lwpImg = rgb_lwpImg.copy()
    H, S, I = cv2.split(hsi_lwpImg)
    for i in range(rows):
        for j in range(cols):
            num = 0.5 * ((r[i, j]-g[i, j])+(r[i, j]-b[i, j]))
            den = np.sqrt((r[i, j]-g[i, j])**2+(r[i, j]-b[i, j])*(g[i, j]-b[i, j]))
            theta = float(np.arccos(num/(den+0.000001)))# epsilon
            
            if den == 0:
                    H = 0
            elif b[i, j] <= g[i, j]:
                H = theta
            else:
                H = 2*3.14169265 - theta

            min_RGB = min(min(b[i, j], g[i, j]), r[i, j])
            sum = b[i, j]+g[i, j]+r[i, j]
            if sum == 0:
                S = 0
            else:
                S = 1 - 3*min_RGB/sum

            H = H/(2*3.14159265)
            I = sum/3.0
            hsi_lwpImg[i, j, 0] = H*255
            hsi_lwpImg[i, j, 1] = S*255
            hsi_lwpImg[i, j, 2] = I*255
    return hsi_lwpImg


def rgb2ycbcr(rgb_lwpImg):
    rows = int(rgb_lwpImg.shape[0])
    cols = int(rgb_lwpImg.shape[1])
    R, G, B = cv2.split(rgb_lwpImg)
    ycbcr_lwpImg = rgb_lwpImg.copy()
    Y, Cb, Cr = cv2.split(ycbcr_lwpImg)
    
    for i in range(rows):
        for j in range(cols):
            
            Y = ( (  66 * R[i,j] + 129 * G[i,j] +  25 * B[i,j] + 128) >> 8) +  16
            Cb = ( ( -38 * R[i,j] -  74 * G[i,j] + 112 * B[i,j] + 128) >> 8) + 128
            Cr = ( ( 112 * R[i,j] -  94 * G[i,j] -  18 * B[i,j] + 128) >> 8) + 128   
    
            ycbcr_lwpImg[i,j,0] = Y
            ycbcr_lwpImg[i,j,1] = Cb
            ycbcr_lwpImg[i,j,2] = Cr
    
    return ycbcr_lwpImg

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
img = cv2.imread('mandrill.ppm')
hsi =  rgb2hsi(img)
ycbcr = rgb2ycbcr(img)

"""
cv2.imwrite("rgb.jpg", img)
cv2.imwrite("hsi.jpg", hsi)
cv2.imwrite("ycbcr.jpg", ycbcr)
"""

# Display Converted Images
cv2.imshow('RGB Image',img)
cv2.imshow('HSI Image',hsi)
cv2.imshow('YCbCr Image',ycbcr)


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

cv2.imshow('Y Channel', ycbcr[:, :, 0])
cv2.imshow('Cr Channel', ycbcr[:, :, 1])
cv2.imshow('Cb Channel', ycbcr[:, :, 2])
"""
# Calculate PSNR Values for HSI and YCbCr with RGB

HSIvalue = PSNR(img, hsi)
print(f"PSNR value between RGB and HSI is {HSIvalue} dB")

YCrCbvalue = PSNR(img, ycbcr)
print(f"PSNR value between RGB and YCbCr is {YCrCbvalue} dB")


# Wait for a key press and then terminate the program
cv2.waitKey(0)
cv2.destroyAllWindows()
