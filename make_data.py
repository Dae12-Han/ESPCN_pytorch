# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 15:51:07 2021

@author: Park
"""

from PIL import Image
from os.path import join, splitext
from os import listdir

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def load_img(filepath):
    img = Image.open(filepath).convert('YCbCr')
    y, _, _ = img.split()
    return y

"""
def main(divisions):
    root_dir = 'dataset/valid'
    image_dir = join(root_dir, "test")
    image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]

    for fname in image_filenames:
        img = Image.open(fname)
        s = splitext(fname)
        
        if divisions == 2: 
            lpart = img.crop((0, 0, img.width/2, img.height))
            rpart = img.crop((img.width/2, 0, img.width, img.height))
        
            lpart.save(s[0]+'_left'+s[1])
            rpart.save(s[0]+'_right'+s[1])
        else:
            ltpart = img.crop((0, 0, img.width/2, img.height/2))
            rtpart = img.crop((img.width/2, 0, img.width, img.height/2))
            lbpart = img.crop((0, img.height/2, img.width/2, img.height))
            rbpart = img.crop((img.width/2, img.height/2, img.width, img.height))
        
            ltpart.save(s[0]+'_lt'+s[1])
            rtpart.save(s[0]+'_rt'+s[1])
            lbpart.save(s[0]+'_lb'+s[1])
            rbpart.save(s[0]+'_rb'+s[1])
"""

def main(xsize, ysize):
    root_dir = 'dataset/Level_Design'
    image_dir = join(root_dir, "original")
    image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]

    for fname in image_filenames:
        img = Image.open(fname)
        s = splitext(fname)
        
        if xsize > img.width or ysize > img.height:
            print(f"{fname}: can't divide the image with {xsize}, {ysize}")
            continue
            
        idx = 0
        xcount = img.width//xsize
        while True:
            xx = idx % xcount
            yy = idx // xcount
            left = xsize*xx
            top = ysize*yy
            right = xsize*(xx+1)
            bottom = ysize*(yy+1)
            
            if right > img.width or bottom > img.height:
                break
            
            part = img.crop((left, top, right, bottom))
            part.save(s[0]+f"_{idx}"+s[1])
            idx += 1

            
if __name__ == '__main__':
    #main(4)   #4등분 or 2등분
    main(256, 256)