import cv2
import numpy as np
from matplotlib import rcParams
import matplotlib.pyplot as plt
rcParams['figure.figsize'] = 100 , 100
rcParams["axes.titlesize"] = 100

def my_SEG(img):
  anyf_out = cv2.fastNlMeansDenoisingColored(img, None, 15, 15, 7, 21)
  hsv_conv = cv2.cvtColor(anyf_out, cv2.COLOR_BGR2HSV)
     
  # Threshold of green in HSV space
  lower_green = np.array([36, 0, 0])
  upper_green = np.array([102, 255, 255])
 
  mask = cv2.inRange(hsv_conv, lower_green, upper_green)
  return mask

img = cv2.imread("iitk.jpg")
img_seg = my_SEG(img)
# cv2.imwrite('hehe.jpg', cv2.bitwise_and(cv2.fastNlMeansDenoisingColored(img, None, 15, 15, 7, 21), cv2.fastNlMeansDenoisingColored(img, None, 15, 15, 7, 21), mask=img_seg))
cv2.imwrite('fordrone.jpg', img_seg)