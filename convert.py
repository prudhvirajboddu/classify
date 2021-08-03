import numpy as np
import cv2 as cv
import os 
    
dir=os.listdir('./train/')

def convert(dir):
    for i in os.listdir('./train/'+dir):
        gif=cv.VideoCapture('./train/'+dir+'/'+i)
        ret,frame=gif.read()
        cv.imwrite('./train/'+dir+'/'+i[:-4]+'.jpg',frame)
        print('./train/'+dir+'/'+i[:-4]+'.jpg')

#after this, you can use convert.py to convert all gifs to jpgs
#importing this file as a module 
#import convert
#convert.convert(dir)
#convert.convert('AD')e.g.
#after conversion remoive the .gif files present using rm *.gif command by changing into specific directory