import time
import cv2
from rtbf import rt_BF
from mybf import my_BF
from table_res import tester

img = cv2.imread("rome.jpg")

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