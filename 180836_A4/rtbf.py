import cv2
import numpy as np
import time

def rt_BF(img_org):
    sigma_s = 16
    sigma_r = 0.075
    kernel_size = 51

    koff = kernel_size // 2

    img = np.zeros((img_org.shape[0] + kernel_size-1, img_org.shape[1] + kernel_size-1, img_org.shape[2]))
    img[koff:img_org.shape[0]+koff,koff:img_org.shape[1]+koff,:] = img_org
    img /= 255
    img = img.astype("float32")

    N = 9
    l = np.linspace(0, 1, N) # Quantization

    gKernel = cv2.getGaussianKernel(kernel_size, sigma_s)
    gKernel = np.matmul(gKernel, gKernel.T)/np.sum(gKernel)**2
    out = np.zeros(img.shape)

    qty_lst = []
    q_lst = []
    for i in range(N):
      im3 = l[i]-img
      im3 = np.exp(-((im3/sigma_r) ** 2)/2)/(2*np.pi*sigma_r**2)
      qty_lst.append(np.divide(cv2.filter2D(np.multiply(img, im3), -1, gKernel), cv2.filter2D(im3, -1, gKernel)))

      if i != N-1:
        q_lst.append((img - l[i])/(l[i+1] - l[i]))

    imgs_ = []
    for i in range(N-1):
      tmp = np.where(l[i] <= img, img, 1e3)
      tmp = np.where(tmp <= l[i+1], (1-q_lst[i])*qty_lst[i] + q_lst[i]*qty_lst[i+1], 0)
      imgs_.append(np.where(tmp <= 1.0, tmp, 0))
    out = np.zeros(img.shape)
    for i in range(N-1):
      out += imgs_[i]

    out = out * 255
    out = np.uint8(out)

    return out[koff:img_org.shape[0]+koff,koff:img_org.shape[1]+koff,:]

if __name__ == "__main__":
    img = cv2.imread("iitk.jpg")
    start = time.time()
    img_rtbf = rt_BF(img)
    cv2.imwrite('rtbf_outputs/iitk_rtbf.jpg', img_rtbf, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    # cv2.imwrite('rtbf_outputs/iitk_rtbf_99.jpg', img_rtbf, [int(cv2.IMWRITE_JPEG_QUALITY), 99])
    # cv2.imwrite('rtbf_outputs/iitk_rtbf_95.jpg', img_rtbf, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    # cv2.imwrite('rtbf_outputs/iitk_rtbf_90.jpg', img_rtbf, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    print("Time taken for implementation:", time.time() - start)