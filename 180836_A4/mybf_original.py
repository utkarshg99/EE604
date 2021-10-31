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

kernel_size = 15
sigma_s = 8
sigma_r = 0.05
gKernel = cv2.getGaussianKernel(kernel_size, sigma_s)
gKernel = np.matmul(gKernel, gKernel.T)/np.sum(gKernel)**2
base_folder = "bf_outputs"
fname = f"{base_folder}/kernelSize_{kernel_size}_sigmaR_{sigma_r}_sigmaS_{sigma_s}".replace(".", "_")+".jpg"

def my_bf(img_org):
  koff = kernel_size // 2

  img = np.zeros((img_org.shape[0] + kernel_size-1, img_org.shape[1] + kernel_size-1, img_org.shape[2]))
  img[koff:img_org.shape[0]+koff,koff:img_org.shape[1]+koff,:] = img_org

  img /= 255
  img = img.astype("float32")
  y, x, z = img.shape

  out = np.copy(img)

  for i in range(koff, y - koff):
    for j in range(koff, x - koff):
      for k in range(z):
        i_q = img[i - koff : i + koff + 1, j - koff : j + koff + 1, k]
        i_p_q = i_q - i_q[koff, koff]
        i_g = np.exp(-((i_p_q/sigma_r) ** 2)/2)/(2*np.pi*sigma_r**2)
        W_ = np.multiply(gKernel, i_g)
        Num_ = np.multiply(i_q, W_)
        out[i, j, k] = np.sum(Num_)/np.sum(W_)

  out *= 255
  out = np.uint8(out)

  return out[koff:img_org.shape[0]+koff,koff:img_org.shape[1]+koff,:]

start_time = time.time()
r_dash = my_bf(rome)                # change between "rome" and "iitk"
plt.subplot(121), plt.imshow(rome)  # change between "rome" and "iitk"
plt.subplot(122), plt.imshow(r_dash)  
plt.show()
print(f"Time Taken: {time.time()-start_time} seconds.")

cv2.imwrite(fname, r_dash[:, :, [2, 1, 0]])
# pil_img = Image.fromarray(r_dash[:, :, [2, 1, 0]])
# pil_img.save(fname.replace("jpg", "raw"))