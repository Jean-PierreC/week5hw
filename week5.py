import numpy as np
import cv2
count = -1
img = cv2.imread("hwimg.png")
cv2.imshow("Original" , img)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
THRESHOLD_MIN = np.array([0,0,0], np.uint8)
THRESHOLD_MAX = np.array([50,30,70], np.uint8)
frame_threshed = cv2.inRange(img_hsv, THRESHOLD_MIN, THRESHOLD_MAX)
img, contours, hierarchy = cv2.findContours(frame_threshed,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, count,(50,55,5),10)
Output_Contour = cv2.approxPolyDP(Input_Contour, epsilon, True)



#void cv2.approxPolyDP(InputArray curve, OutputArray)
#approxCurve, double epsilon, bool closed)
    for i in contours:
        approx= cv2.approxpolyDP(i, epsilon, etc.)
        if (approx = 4 ,.......):
            maxCosine = 0
            for k in range(2,5):
                pt1 = approx[k%4]
                pt2 = approx[k-2]
                pt0 = approx[k-1]
                cos = (angle(pt1, pt2, pt0))
                cosine = math.fabs(math.cos(cos))
                def angle(p1, p2, p0):
                    dx1 = p1[0] [0] - p0[0] [0]
                    dy1 = p1[0] [1] - p0[0] [1]
                    dx2 = p2[0] [0] - p0[0] [0]
                    dy2 = p2[0] [1] - p0[0] [1]
                    return mat.atan(dy1/dx1)-math.atan(dy2/dx2)
double cv2.contourArea(InputArray contour)
bool cv2.isContourConvex(InputArray contour)
for every contour:
    cv2.approxpolyDP(current_contour, approx, etc.)
    if (approx has four......):
        cv2.imshow("week5",img)
        cv2.waitKey(0)

