import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
import csv



# Define characters
Characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghiijjklmnopqrstuvwxyz'
Characters = 'AathUHZOnbBuVPIJjocCJVWKQipDdwkRLXEqeXMlSrYFfymTZNgSg'
#print(len(Characters))
#read image
img = plt.imread('HelloWorld.jpg', format=None)
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


#plt.imshow(img_gray,cmap='gray')
#plt.figure()

ret,thresh = cv2.threshold(img_gray,127,1,cv2.THRESH_BINARY_INV)
height = thresh.shape[0]
width = thresh.shape[1]


plt.imshow(thresh,cmap='gray')
plt.figure()
lines = []

def seg(thresh):

    start_line = False

    for row in range(thresh.shape[0]):
        blank = True
        if thresh[row].sum() != 0:
            blank = False
            #print("Line is not blank")
        
        if blank == True:
            #print("skip line")
            if start_line != False:
                Image1 = thresh[start_line:row]
                lines.append([start_line,row])
                #Image11 = Image1.shape[1]
                plt.imshow(Image1,cmap='gray')
                plt.figure()
                start_line = False
        else:
            if start_line == False:
                start_line = row
           #print("do not skip line")
    char = []
    for line in lines:
        Char_found = False
       #print(line)
        for col in range(thresh.shape[1]):
            blank = True
            if thresh[line[0]:line[1],col].sum() != 0:
                blank =False
            if blank == True:
                if Char_found != False:
                      char.append([line[0],line[1],Char_found,col]) 
                      
                      Image1 = thresh[line[0]:line[1],Char_found:col]
                      plt.imshow(Image1,cmap='gray')
                      plt.figure()
                      Char_found = False
            else:
                if Char_found == False:
                    Char_found = col
    letter = []
    for Character in char:
       # print(Character)
        start_line = False

        for row in range(Character[0],Character[1]+1):
           #print(row)
            blank = True
            if thresh[row,Character[2]:Character[3]].sum() != 0:
                blank = False
               # print("Line is not blank")
        
            if blank == True:
                #print("skip line")
                if start_line != False:
                    #print(start_line,row,Character[2],Character[3])
                    letter.append([start_line,row,Character[2],Character[3]])
                    #img1 = thresh[start_line:row,Character[2]:Character[3]]
                   #lines.append([start_line,row])
                    #Image11 = Image1.shape[1]
                    #plt.imshow(img1,cmap='gray')
                    #plt.figure()
                    start_line = False
            else:
                if start_line == False:
                    start_line = row
                    #print(start_line)
                if row == Character[1] and start_line != False:
                    #print('yesh')
                    img1 = thresh[start_line:row,Character[2]:Character[3]]
                    letter.append([start_line,row,Character[2],Character[3]])
                    plt.imshow(img1,cmap='gray')
                    plt.figure()
    return letter

segments = seg(thresh)

for seg in segments:
    img = thresh[seg[0]:seg[1],seg[2]:seg[3]]
    #cv2.imwrite(str(seg) + '.jpg' , img)
   # plt.imshow(img,cmap='gray')
    #plt.figure()

    new_height1 = 64
    new_width1 = 64
    new_image = cv2.resize(img ,(new_height1,new_width1))
    #thresh1 = new_image
    #ret1,thresh1 = cv2.threshold(new_image,127,1,1)
   # print(thresh1)
    #plt.imshow(thresh1,cmap='gray')
    #plt.figure()

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
           # print(A)
            
    Walsh_Image = A * new_image * A
    
    l = np.array(Walsh_Image)
    #print(l)
    
        
    guess = '?'
    distance = 100000
    
    #print(l[:,1])
    #plt.imshow(Walsh_Image,cmap = 'gray')
    #plt.show()
    
    with open('DataSet.csv', 'r') as csvfile:
        spamreader = csv.reader(csvfile)
        count = 0
        for row in spamreader:
            letter = np.array(row)
            letter = letter.reshape(64,64)
            letter = letter.astype(np.float16)
            dist = np.linalg.norm(letter-l)
           # print(dist)
            if dist < distance:
                guess = Characters[count]
                distance = dist
            count = count + 1
           # print(l[:1])
    print(guess)
            