import cv2
import numpy as np
import math, sys, time
import matplotlib.pyplot as plt
from matplotlib import rcParams
from skimage.metrics import structural_similarity
from PIL import Image
rcParams['figure.figsize'] = 100 , 100
rcParams["axes.titlesize"] = 100

original_img_f = 'iitk.jpg'
final_img_f = 'bf_outputs_new/iitk.jpg'
original_img = cv2.imread(original_img_f)
final_img = cv2.imread(final_img_f)

def PSNR(original, filtered):
    mse = np.mean((original - filtered) ** 2)
    if(mse == 0):
        return 100
    max_pixel = 255.0
    return 20 * math.log10(max_pixel / math.sqrt(mse))

psnr = PSNR(original_img, final_img)

(score, diff) = structural_similarity(original_img, final_img, full=True,  multichannel = True)
diff = (diff * 255).astype("uint8")

print(f"Files Compared: {original_img_f} vs {final_img_f}")
print(f"SSIM: {score}, PSNR: {psnr}dB")