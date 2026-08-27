import cv2
import numpy as np


left = cv2.imread(r"C:\Users\abdul\OneDrive\Desktop\task9 cv\im0.png", cv2.IMREAD_GRAYSCALE)
right = cv2.imread(r"C:\Users\abdul\OneDrive\Desktop\task9 cv\im1.png", cv2.IMREAD_GRAYSCALE)

stereo  = cv2.StereoBM_create( numDisparities= 288, blockSize=15)

disparity = stereo.compute(left,right)

disp_norm = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)


cv2.namedWindow("Left", cv2.WINDOW_NORMAL)
cv2.namedWindow("Right", cv2.WINDOW_NORMAL)
cv2.namedWindow("Disparity", cv2.WINDOW_NORMAL)

cv2.resizeWindow("Left", 800, 600)
cv2.resizeWindow("Right", 800, 600)
cv2.resizeWindow("Disparity", 800, 600)

cv2.imshow("Left", left)
cv2.imshow("Right", right)
cv2.imshow("Disparity", disp_norm)
cv2.waitKey(0)
cv2.destroyAllWindows()