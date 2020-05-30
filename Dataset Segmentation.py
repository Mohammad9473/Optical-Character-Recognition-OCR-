# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 14:43:30 2019

@author: Mohammadreza
"""

import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
import csv

#read image
img = cv2.imread('Dataset.jpg')

#grayscale
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.waitKey(0)





#binarize 
ret,thresh = cv2.threshold(gray,127,1,cv2.THRESH_BINARY_INV)
cv2.waitKey(0)
#cv2.THRESH_BINARY_INV
#find contours

dimensions = gray.shape
height = gray.shape[0]
width = gray.shape[1]

#print(height)

im2,ctrs, hier = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, 
cv2.CHAIN_APPROX_SIMPLE)

#sort contours
sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

for i, ctr in enumerate(sorted_ctrs):
    # Get bounding box
    x, y, w, h = cv2.boundingRect(ctr)

    # Getting ROI
    roi = gray[y:y+h, x:x+w]

    cv2.waitKey(0)
    new_height1 = 64
    new_width1 = 64
    new_image = cv2.resize(roi ,(new_height1,new_width1))
    cv2.imwrite(str(i) + '.jpg', new_image)
    cv2.rectangle(gray,(x,y),( x + w, y + h ),(90,0,255),2)
    ret1,thresh1 = cv2.threshold(new_image,127,1,1)
    
    new_image = thresh1


    #cv2.imshow('window', new_image)
  


    A = np.zeros((new_height1,new_width1))
    n = int(math.log(new_width1,2))
   # print(n)
    for e in range(new_width1):
        for j in range(new_height1):
            P = 1
            P_bin = bin(e)[2:].zfill(n)
            P_bin1 = bin(j)[2:].zfill(n)
            for v in range(n):
                D = int(P_bin[v*-1-1]) * int(P_bin1[v])
                E = int(np.power(-1,D))
                P = P * E
            A[e][j] = P
    Walsh_Image = A * new_image * A
    
    l = np.array(Walsh_Image)
    print(l)
    #l1 = l[:,1]
    with open('DataSet.csv', 'a', newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(Walsh_Image.flatten())
    print(l[:,1])
    #plt.imshow(Walsh_Image,cmap = 'gray')
    #plt.show()
    