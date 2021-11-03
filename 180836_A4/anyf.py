import cv2
import numpy as np
import math, sys, time
import matplotlib.pyplot as plt
from matplotlib import rcParams
from PIL import Image
rcParams['figure.figsize'] = 100 , 100
rcParams["axes.titlesize"] = 100

rome = cv2.imread('rome.jpg')
iitk = cv2.imread('iitk.jpg')
rome = rome[:,:,[2,1,0]]
iitk = iitk[:,:,[2,1,0]]

p1 = 10 # parameter deciding filter strength
p2 = p1 # same as p1, but for colored images
p3 = 7  # templateWindowSize [Should be odd, recommended: 7]
p4 = 15 # searchWindowSize [Should be odd, recommended: 21]
base_folder = "anyf_outputs"
fname = f"{base_folder}/h_{p1}_tempWS_{p3}_srchWS_{p4}".replace(".", "_")+".jpg"

start_time = time.time()
r_dash = cv2.fastNlMeansDenoisingColored(rome, None, p1, p2, p3, p4)  # change between "rome" and "iitk"
plt.subplot(121), plt.imshow(rome)                                   # change between "rome" and "iitk"
plt.subplot(122), plt.imshow(r_dash)  
plt.show()
print(f"Time Taken: {time.time()-start_time} seconds.")

cv2.imwrite(fname, r_dash[:, :, [2, 1, 0]])