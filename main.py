import cv2
import numpy as np




left = cv2.imread(r"C:\Users\abdul\OneDrive\Desktop\task9 cv\im0.png",cv2.IMREAD_GRAYSCALE)

right = cv2.imread(r"C:\Users\abdul\OneDrive\Desktop\task9 cv\im1.png", cv2.IMREAD_GRAYSCALE)





stereo = cv2.StereoBM_create( numDisparities=288, blockSize=15)



disparity = stereo.compute(left, right)

disparity = disparity.astype(np.float32) / 16.0




disp_norm = cv2.normalize(disparity, None, 0,255,cv2.NORM_MINMAX)

disp_norm = disp_norm.astype(np.uint8)



focal_length = 979.911

base_line = 193.001 / 1000.0




depth = np.zeros_like(disparity,dtype=np.float32)

valid = disparity > 0

depth[valid] = (focal_length * base_line) / disparity[valid]



x = 500
y = 300

d = disparity[y, x]

if d > 0:

    z = depth[y, x]

    print("Disparity =", d)
    print("Depth =", z, "meters")

else:

    print("Invalid disparity at this point")




cv2.namedWindow("Left", cv2.WINDOW_NORMAL)
cv2.namedWindow("Right", cv2.WINDOW_NORMAL)
cv2.namedWindow("Disparity", cv2.WINDOW_NORMAL)

cv2.resizeWindow("Left", 800, 600)
cv2.resizeWindow("Right", 800, 600)
cv2.resizeWindow("Disparity", 800, 600)

cv2.imshow("Left", left)
cv2.imshow("Right", right)
cv2.imshow("Disparity", disp_norm)
cv2.imwrite("image.png", disparity)

cv2.waitKey(0)
cv2.destroyAllWindows()