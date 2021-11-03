import cv2
import numpy as np
import math
from skimage.metrics import structural_similarity

def tester(original_img, final_img):

    def PSNR(original, filtered):
        mse = np.mean((original - filtered) ** 2)
        if(mse == 0):
            return 100
        max_pixel = 255.0
        return 20 * math.log10(max_pixel / math.sqrt(mse))

    psnr = PSNR(original_img, final_img)

    (score, diff) = structural_similarity(original_img, final_img, full=True,  multichannel = True)
    diff = (diff * 255).astype("uint8")
    print(f"SSIM: {score}\t PSNR: {psnr}dB")