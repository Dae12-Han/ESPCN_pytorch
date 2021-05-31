# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 14:13:33 2021

@author: Park
"""

from PIL import Image
from data import calculate_valid_crop_size
from dataset import is_image_file
from os import listdir
from os.path import join
import numpy as np
from math import log10
from torchvision.transforms import Compose, ToTensor
import torch.nn as nn

upscale_factor = 3


def center_crop(im, new_width, new_height):
    width, height = im.size   # Get dimensions
    left = (width - new_width)//2
    top = (height - new_height)//2
    right = (width + new_width)//2
    bottom = (height + new_height)//2

    # Crop the center of the image
    out = im.crop((left, top, right, bottom))
    return out
    

def normalize(arr):
    """
    Linear normalization
    http://en.wikipedia.org/wiki/Normalization_%28image_processing%29
    """
    arr = arr.astype('float')
   
    minval = arr[...].min()
    maxval = arr[...].max()
    if minval != maxval:
        #arr[...] -= minval
        #arr[...] *= (1.0/(maxval-minval))
        print(maxval)
        arr[...] *= 1.0/255
    return arr

def calc_psnr(root_dir):
    image_dir = join(root_dir, "test")
    crop_size = calculate_valid_crop_size(256, upscale_factor)
    
    image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
    
    avg_psnr = 0
    for filepath in image_filenames:
        img = Image.open(filepath).convert('YCbCr')
        y, _, _ = img.split()
                        
        y = center_crop(y, crop_size, crop_size)
        small = y.resize((y.size[0]//upscale_factor, y.size[1]//upscale_factor), Image.BILINEAR)
        large = small.resize((small.size[0]*upscale_factor, small.size[1]*upscale_factor), Image.BICUBIC)
        #large.show()
        #y.show()
        target = np.array(y)
        prediction = np.array(large)
        
        transform = Compose([
            ToTensor()])
        ty = transform(y)
        tp = transform(prediction)
        mse = nn.MSELoss()(ty, tp)
        psnr = 10 * log10(1 / mse.item())
       
        #sqerr = np.sum((normalize(target) - normalize(prediction))**2)
        #mse = sqerr/float(target.shape[0]*target.shape[1])
        #psnr = 10 * log10(1 / mse)
        avg_psnr += psnr
    
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(image_filenames)))


def main():
    calc_psnr('dataset/CG')
    

if __name__ == '__main__':
    main()