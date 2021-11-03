import cv2
import numpy as np
import time

def my_BF(img_org, stride = 1, fromMain = None):

  kernel_size = 51
  sigma_s = 16
  sigma_r = 0.075
  # stride = 2      # increasing this value will decrease the time taken at the cost of quality
  gKernel = cv2.getGaussianKernel(kernel_size, sigma_s)
  gKernel = np.matmul(gKernel, gKernel.T)/np.sum(gKernel)**2
  global fname
  if fromMain:
    fname = f"{base_folder}/kernelSize_{kernel_size}_sigmaR_{sigma_r}_sigmaS_{sigma_s}_Stride_{stride}".replace(".", "_")+".jpg"

  koff = kernel_size // 2

  img = np.zeros((img_org.shape[0] + kernel_size-1, img_org.shape[1] + kernel_size-1, img_org.shape[2]))
  img[koff:img_org.shape[0]+koff,koff:img_org.shape[1]+koff,:] = img_org

  img /= 255
  img = img.astype("float32")
  y, x, z = img.shape

  out = np.copy(img)

  for i in range(koff, y - koff, stride):
    for j in range(koff, x - koff, stride):
      for k in range(z):
        i_q = img[i - koff : i + koff + 1, j - koff : j + koff + 1, k]
        i_p_q = i_q - i_q[koff, koff]
        i_g = np.exp(-((i_p_q/sigma_r) ** 2)/2)/(2*np.pi*sigma_r**2)
        W_ = np.multiply(gKernel, i_g)
        Num_ = np.multiply(i_q, W_)
        out[i - stride//2 : i + stride//2 + 1, j - stride//2 : j + stride//2 + 1, k] = np.sum(Num_)/np.sum(W_)

  out = out * 255
  out = np.uint8(out)

  return out[koff:img_org.shape[0]+koff,koff:img_org.shape[1]+koff,:]

if __name__ == "__main__":
  img_l = cv2.imread('iitk.jpg')
  img_l = img_l[:,:,[2,1,0]]
  base_folder = "bf_outputs_new"
  fname = "generic.jpg"
  start_time = time.time()
  r_dash = my_BF(img_l, 2, True)
  print(f"Time Taken: {time.time()-start_time} seconds.")

  cv2.imwrite(fname, r_dash[:, :, [2, 1, 0]])