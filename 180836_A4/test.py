import time
import cv2
from rtbf import rt_BF
from mybf import my_BF
from table_res import tester

img = cv2.imread("iitk.jpg")

_i = 1
s = time.time()
img_mybf = my_BF(img)
img_rtbf = rt_BF(img)
print(f"\nTest #{_i}\tTime Taken: {time.time()-s}")
tester(img_mybf, img_rtbf)

_i += 1
s = time.time()
img_rtbf = rt_BF(img)
cv2.imwrite('temp.jpg', img_rtbf, [int(cv2.IMWRITE_JPEG_QUALITY), 99])
img_rtbf_99 = cv2.imread("temp.jpg")
print(f"\nTest #{_i}\tTime Taken: {time.time()-s}")
tester(img_mybf, img_rtbf)

_i += 1
s = time.time()
img_rtbf = rt_BF(img)
cv2.imwrite('temp.jpg', img_rtbf, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
img_rtbf_95 = cv2.imread("temp.jpg")
print(f"\nTest #{_i}\tTime Taken: {time.time()-s}")
tester(img_mybf, img_rtbf_95)

_i += 1
s = time.time()
img_rtbf = rt_BF(img)
cv2.imwrite('temp.jpg', img_rtbf, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
img_rtbf_90 = cv2.imread("temp.jpg")
print(f"\nTest #{_i}\tTime Taken: {time.time()-s}")
tester(img_mybf, img_rtbf_90)


# img_y = cv2.imread("your.jpg")
# def edge_mask(img, line_size, blur_value):
#   gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#   gray_blur = cv2.medianBlur(gray, blur_value)
#   edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_size, blur_value)
#   return edges
# edges = edge_mask(img_y, 9, 9)

# img_rtbf_crtn = rt_BF(img_y)
# img_rtbf_crtn = rt_BF(img_rtbf_crtn)
# img_rtbf_crtn = rt_BF(img_rtbf_crtn)
# img_rtbf_crtn = rt_BF(img_rtbf_crtn)
# img_rtbf_crtn = rt_BF(img_rtbf_crtn)

# print(img_y.shape)
# print(edges.shape)
# cartoon = cv2.bitwise_and(img_rtbf_crtn, img_rtbf_crtn, mask=edges)

# cv2.imwrite("cartoon.jpg", img_rtbf_crtn)