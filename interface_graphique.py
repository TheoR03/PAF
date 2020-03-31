#!/usr/bin/env python
# -*- coding: utf-8 -*-
#include <opencv2/core/ocl.hpp>


import os,cv2
import PIL.Image, PIL.ImageTk
import numpy as np
from tkinter import *
from tkinter.font import Font,BOLD
from tkinter.ttk import Style,Treeview
import numpy as np
import cv2
from math import sqrt
import copy
import scipy.spatial
import math
from tkinter.messagebox import *
from win32api import GetSystemMetrics
from shutil import copyfile


from tkinter.filedialog import askopenfilename,askdirectory
import numpy as np
import math
import cv2
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from joblib import dump, load
import pickle

from tempfile import mkdtemp
savedir = mkdtemp()
import os


def from255To1(mask):
    (lineNumber,columnNumber)=mask.shape
    mask = (mask/np.max(mask)).astype('float')
    return mask

def compare(initialMask,transformedMask):
    (lineNumber,columnNumber)=initialMask.shape

    counter = np.sum(np.abs(initialMask - transformedMask )).astype(int)
    return counter#/area
    
##ASSYMETRIE

def centroid(thresh):
    M = cv2.moments(thresh)

# calculate x,y coordinate of center
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return(cX,cY)

def distance(x1,y1,x2,y2):#Calcule la distance entre deux points
    return math.sqrt((x2-x1)**2+(y2-y1)**2)

def angle(xc,yc,x,y):
    cos=distance(x,yc,xc,yc)/distance(xc,yc,x,y)
    return math.acos(cos)

def maxDistancePoint(mask): #Renvoie les coordonnées du point du grain de beauté le plus éloigné du centroïde
    (xc,yc)=centroid(mask)
    (lineNumber,columnNumber)=mask.shape
    xMax=0
    yMax=0
    distanceMax=0
    dist=0.0

    for i in range(lineNumber):
        for j in range(columnNumber):
            if(mask[i][j]==1):
                dist=distance(xc,yc,i,j)
                if(dist>distanceMax):
                    xMax=j
                    yMax=i
                    distanceMax=dist
    return(xMax,yMax)



def baseCoord(mask):
    (xc,yc)=centroid(mask)
    (x1,y1)=maxDistancePoint(mask)
    vector1=(x1-xc,y1-yc)
    xa=vector1[0]
    ya=vector1[1]
    x1=200
    vector2=(x1,-x1*(xa/ya))
    return(vector1,vector2)



def rotate_image(mat, angle,xc,yc):
    height, width = mat.shape
    image_center = (xc, yc)

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1)
    rotated_mat = cv2.warpAffine(mat.astype(float), rotation_mat, (width,height) )
    return rotated_mat

def assymmetry(mask): #Le masque doit être binaire
    (xc,yc)=centroid(mask)
    rotatedMask=rotate_image(mask,180,xc,yc)
    return compare(mask,rotatedMask)

def mirror_principalAxis(mask):
    (xc,yc)=centroid(mask)
    
    # trouver axes principaux
    #extraire coordonnees des elements non-nuls dans mask
    y_inds, x_inds = np.nonzero(mask>0)
    y_inds = y_inds - yc
    x_inds = x_inds - xc
    y_carre = (y_inds**2).sum()
    x_carre = (x_inds**2).sum()
    y_x = (np.multiply(y_inds,x_inds)).sum()

    A = np.zeros((2,2))
    A[0,0] = y_carre
    A[0,1] = y_x
    A[1,0] = y_x
    A[1,1] = x_carre
    w,v = np.linalg.eig(A)
    
    if abs(w[0])>abs(w[1]):
        theta = angle(xc,yc,xc+v[1,0]*800,yc+v[0,0]*800)
    else:
        theta = angle(xc,yc,xc+v[1,1]*800,yc+v[0,1]*800)
    mask=rotate_image(mask,-theta*180/math.pi,xc,yc) 
    mask=np.flipud(mask)
    mask=rotate_image(mask,theta*180/math.pi,xc,yc)
   
    return mask

def applyMask(img,mask):
    (x,y)=mask.shape
    for i in range (x):
        for j in range (y):
            if mask[i][j]==0:
                img[i][j]=[0,0,0]
    return img

#%%ABCD

def ABCD(imageName,maskName):
    image=cv2.imread(imageName,1)
    mask = cv2.imread(maskName,0)

    (x,y)=mask.shape
    list=[0,0,0,0,0]

    #assymetry

    newDim = max(x,y)
    enlargedMask= np.zeros((2*newDim,2*newDim))
    enlargedMask[0:x,0:y] = mask
    enlargedMask = np.roll(enlargedMask,(newDim-x//2-1,newDim-y//2-1),(0,1))
    enlargedMask=from255To1(enlargedMask)
    list[0]=assymmetry(enlargedMask)
    
    list[1]=compare(enlargedMask,mirror_principalAxis(enlargedMask))

    #similarToEllipse
    ret,thresh = cv2.threshold(mask,127,255,0)
    contours,hierarchy = cv2.findContours(thresh, 1, 2) #contours,hierarchy
    cnt = contours[0]
    ellipse = cv2.fitEllipse(cnt)
    ellipseMask=np.zeros((x,y))
    cv2.ellipse(ellipseMask,ellipse,255,-1)
    list[2]=compare(from255To1(mask),from255To1(ellipseMask))

    #perimeter/area
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt,True)
    list[3]=perimeter/area*1000

    #color
    binaryMask=from255To1(mask).astype(np.uint8)
    
    hist = (cv2.calcHist([image], [0], binaryMask, [256], [0, 255])+cv2.calcHist([image], [1], binaryMask, [256], [0, 255])+cv2.calcHist([image], [2], binaryMask, [256], [0, 255]))/3
   
    list[4]=np.var(hist)

    return np.array(list).reshape(1,-1)

def ABCDbis(imageName,mask):
    image=cv2.imread(imageName,1)

    (x,y)=mask.shape
    list=[0,0,0,0,0]

    #assymetry

    newDim = max(x,y)
    enlargedMask= np.zeros((2*newDim,2*newDim))
    enlargedMask[0:x,0:y] = mask
    enlargedMask = np.roll(enlargedMask,(newDim-x//2-1,newDim-y//2-1),(0,1))
    enlargedMask=from255To1(enlargedMask)
    list[0]=assymmetry(enlargedMask)
    
    list[1]=compare(enlargedMask,mirror_principalAxis(enlargedMask))

    #similarToEllipse
    ret,thresh = cv2.threshold(mask,127,255,0)
    contours,hierarchy = cv2.findContours(thresh, 1, 2) #contours,hierarchy
    cnt = contours[0]
    ellipse = cv2.fitEllipse(cnt)
    ellipseMask=np.zeros((x,y))
    cv2.ellipse(ellipseMask,ellipse,255,-1)
    list[2]=compare(from255To1(mask),from255To1(ellipseMask))

    #perimeter/area
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt,True)
    list[3]=perimeter/area*1000

    #colour
    binaryMask=from255To1(mask).astype(np.uint8)
    
    hist = (cv2.calcHist([image], [0], binaryMask, [256], [0, 255])+cv2.calcHist([image], [1], binaryMask, [256], [0, 255])+cv2.calcHist([image], [2], binaryMask, [256], [0, 255]))/3
   
    list[4]=np.var(hist)

    return np.array(list).reshape(1,-1)

#%%TEST CLASSIFICATION - PRE PROCESSING

def preProcessing(address):

    df = pd.read_excel(address)
    df.as_matrix()
    # Getting the data from excel into a python data stream

    features1 = np.array(df)
    # Data inside a matrix --> ground truth about the nature of  spots

    features = []
    for k in range(len(features1)):
        if features1[k][2]==0:
            features.append(features1[k])
            #get rid of seborrheic keratosis lines

    features = np.array(features)
    features = features[::,:2]
    # 0 means benign, 1 means malignant
    return features

#%%TEST CLASSIFICATION - LDA

excelAddress = 'C:\\Users\\theor\\Downloads\\PAF\\Ground_truth_ISIC_1.xlsx'
trainingSetLength = 5

diagnostic=preProcessing(excelAddress)
(length,columnNumber)=diagnostic.shape
extractedFeatures=np.zeros((length,5))

for i in range(trainingSetLength):#previous version range(lenght)

    imageAddress='C:\\Users\\theor\\Downloads\\PAF\\ISIC-2017_Training_Data\\ISIC-2017_Training_Data\\'+diagnostic[i,0]+'.jpg'
    maskAddress='C:\\Users\\theor\\Downloads\\PAF\\Masks\\'+diagnostic[i,0]+'_segmentation.png'
    extractedFeatures[i]=ABCD(imageAddress,maskAddress)

y = diagnostic[:trainingSetLength,1:] # target values (i.e. expected output for X)

for i in range (len(y)):
    y[i]=int(y[i])
y=np.transpose(y).astype('int')
    
trainingSet=extractedFeatures[:trainingSetLength]
    
lda = LinearDiscriminantAnalysis()
    
lda.fit(trainingSet, y[0])
# letting the algorithm know which sample in X belongs to which class labelled in y
    
# save the params to disk
lda_params = lda.get_params()
params_lda = 'params_lda.sav'

# save the model to disk
filename_lda = 'lda_model.sav'
    
    
#testSet=extractedFeatures[trainingSetLength:trainingSetLength+10]
#prediction=lda.predict(testSet)

pickle.dump(lda, open(filename_lda, 'wb'))
pickle.dump(lda_params, open(params_lda, 'wb'))
    
#%%TEST CLASSIFICATION - QDA

excelAddress = 'C:\\Users\\theor\\Downloads\\PAF\\Ground_truth_ISIC_1.xlsx'
trainingSetLength = 5

diagnostic=preProcessing(excelAddress)
(length,columnNumber)=diagnostic.shape
extractedFeatures=np.zeros((length,5))

for i in range(trainingSetLength):#previous version range(lenght)
    
    imageAddress='C:\\Users\\theor\\Downloads\\PAF\\ISIC-2017_Training_Data\\ISIC-2017_Training_Data\\'+diagnostic[i,0]+'.jpg'
    maskAddress='C:\\Users\\theor\\Downloads\\PAF\\Masks\\'+diagnostic[i,0]+'_segmentation.png'
    extractedFeatures[i]=ABCD(imageAddress,maskAddress)

y = diagnostic[:trainingSetLength,1:] # target values (i.e. expected output for X)

for i in range (len(y)):
    y[i]=int(y[i])
y=np.transpose(y).astype('int')

trainingSet=extractedFeatures[:trainingSetLength]
qda = QuadraticDiscriminantAnalysis()

qda.fit(trainingSet, y[0])
# letting the algorithm know which sample in X belongs to which class labelled in y
        
# save the params to disk
qda_params = qda.get_params()
params_qda = 'params_qda.sav'
    
# save the model to disk
filename_qda = 'qda_model.sav'
pickle.dump(qda, open(filename_qda, 'wb'))
    
#testSet=extractedFeatures[trainingSetLength:trainingSetLength+10]
#prediction=qda.predict(testSet)

pickle.dump(qda_params, open(params_qda, 'wb'))
#%%TEST CLASSIFICATION - Naive Bayes
excelAddress = 'C:\\Users\\theor\\Downloads\\PAF\\Ground_truth_ISIC_1.xlsx'
trainingSetLength = 5

diagnostic=preProcessing(excelAddress)
(length,columnNumber)=diagnostic.shape
extractedFeatures=np.zeros((length,5))

for i in range(trainingSetLength):#previous version range(lenght)

    imageAddress='C:\\Users\\theor\\Downloads\\PAF\\ISIC-2017_Training_Data\\ISIC-2017_Training_Data\\'+diagnostic[i,0]+'.jpg'
    maskAddress='C:\\Users\\theor\\Downloads\\PAF\\Masks\\'+diagnostic[i,0]+'_segmentation.png'
    extractedFeatures[i]=ABCD(imageAddress,maskAddress)

y = diagnostic[:trainingSetLength,1:] # target values (i.e. expected output for X)

for i in range (len(y)):
    y[i]=int(y[i])
y=np.transpose(y).astype('int')

trainingSet=extractedFeatures[:trainingSetLength]
bys = GaussianNB()

bys.fit(trainingSet, y[0])
# letting the algorithm know which sample in X belongs to which class labelled in y
    
# save the params to disk
bys_params = bys.get_params()
params_bys = 'params_bys.sav'
    
# save the model to disk
filename_bys = 'bys_model.sav'
pickle.dump(bys, open(filename_bys, 'wb'))
    
#testSet=extractedFeatures[trainingSetLength:trainingSetLength+10]
#prediction=bys.predict(testSet)

pickle.dump(bys_params, open(params_bys, 'wb'))
#%%TEST CLASSIFICATION - kNN
excelAddress = 'C:\\Users\\theor\\Downloads\\PAF\\Ground_truth_ISIC_1.xlsx'
trainingSetLength = 5

diagnostic=preProcessing(excelAddress)
(length,columnNumber)=diagnostic.shape
extractedFeatures=np.zeros((length,5))

for i in range(trainingSetLength):#previous version range(lenght)

    imageAddress='C:\\Users\\theor\\Downloads\\PAF\\ISIC-2017_Training_Data\\ISIC-2017_Training_Data\\'+diagnostic[i,0]+'.jpg'
    maskAddress='C:\\Users\\theor\\Downloads\\PAF\\Masks\\'+diagnostic[i,0]+'_segmentation.png'
    extractedFeatures[i]=ABCD(imageAddress,maskAddress)

y = diagnostic[:trainingSetLength,1:] # target values (i.e. expected output for X)

for i in range (len(y)):
    y[i]=int(y[i])
y=np.transpose(y).astype('int')

trainingSet=extractedFeatures[:trainingSetLength]
neigh = KNeighborsClassifier(n_neighbors=1)

neigh.fit(trainingSet, y[0])
# letting the algorithm know which sample in X belongs to which class labelled in y
    
# save the params to disk
neigh_params = neigh.get_params()
params_neigh = 'params_neigh.sav'
    
# save the model to disk
filename_neigh = 'neigh_model.sav'
pickle.dump(neigh, open(filename_neigh, 'wb'))
    
#testSet=extractedFeatures[trainingSetLength:trainingSetLength+10]
#prediction=lda.predict(testSet)

pickle.dump(neigh_params, open(params_neigh, 'wb'))
#%%TEST CLASSIFICATION - Logistic Regression
excelAddress = 'C:\\Users\\theor\\Downloads\\PAF\\Ground_truth_ISIC_1.xlsx'
trainingSetLength = 5

diagnostic=preProcessing(excelAddress)
(length,columnNumber)=diagnostic.shape
extractedFeatures=np.zeros((length,5))

for i in range(trainingSetLength):#previous version range(lenght)

    imageAddress='C:\\Users\\theor\\Downloads\\PAF\\ISIC-2017_Training_Data\\ISIC-2017_Training_Data\\'+diagnostic[i,0]+'.jpg'
    maskAddress='C:\\Users\\theor\\Downloads\\PAF\\Masks\\'+diagnostic[i,0]+'_segmentation.png'
    extractedFeatures[i]=ABCD(imageAddress,maskAddress)

y = diagnostic[:trainingSetLength,1:] # target values (i.e. expected output for X)

for i in range (len(y)):
    y[i]=int(y[i])
y=np.transpose(y).astype('int')

trainingSet=extractedFeatures[:trainingSetLength]
lgr = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')

lgr.fit(trainingSet, y[0])
# letting the algorithm know which sample in X belongs to which class labelled in y
    
# save the params to disk
lgr_params = lgr.get_params()
params_lgr = 'params_lgr.sav'
    
# save the model to disk
filename_lgr = 'lgr_model.sav'
pickle.dump(lgr, open(filename_lgr, 'wb'))
    
#testSet=extractedFeatures[trainingSetLength:trainingSetLength+10]
#prediction=lda.predict(testSet)
   
pickle.dump(lgr_params, open(params_lgr, 'wb'))  

#%%TEST CLASSIFICATION - Diagnostic

def diagnostic(imageAddress,mask):
    features=ABCDbis(imageAddress,mask)
    diagnostics=[0,0,0,0]
    
    lda = pickle.load(open(filename_lda, 'rb'))
    paramsLda=pickle.load(open(params_lda,'rb'))
    lda.set_params(**paramsLda)
    diagnostics[0]=lda.predict(features).tolist()[0]
    
    qda = pickle.load(open(filename_qda, 'rb'))
    paramsQda=pickle.load(open(params_qda,'rb'))
    qda.set_params(**paramsQda)
    diagnostics[1]=qda.predict(features).tolist()[0]
    
    bys = pickle.load(open(filename_bys, 'rb'))
    paramsBys=pickle.load(open(params_bys,'rb'))
    bys.set_params(**paramsBys)
    diagnostics[2]=bys.predict(features).tolist()[0]
    
    lgr = pickle.load(open(filename_lgr, 'rb'))
    paramsLr=pickle.load(open(params_lgr,'rb'))
    lgr.set_params(**paramsLr)
    diagnostics[3]=lgr.predict(features).tolist()[0]
    
    return diagnostics


address = 'C:\\Users\\theor\\Downloads\\PAF\\Ground_truth_ISIC_1.xlsx'
#print(linearDiscriminantAnalysis(address,10))


#%% How to find current working directory (where pickle save files are stored)

import os 
dir_path = os.path.dirname(os.path.realpath(filename_lda))
print(dir_path)

#%% How to load the model from a pickle save file

lda_bis = pickle.load(open(filename_lda, 'rb'))
print(lda_bis)
##

def ColorSpaceTransformation(img_in):
    #We keep this transformation for the moment : the extraction of the blue channel 
    img_BaW = img_in[:,:,0]
    
    return img_BaW


#return a blurred image using a parameter for the blur amount
def noiseRemoval(img):
    blur = 5
    kernel = np.ones((blur, blur), np.float32) / (blur**2)
    dst = cv2.filter2D(img, -1, kernel)

    return(dst)

#Map the values of the intensity from the current min and max to 0 and 255 with a certain percentage of saturation
def intensityAdjust(imgBaW):

    minOut = 0
    maxOut = 255

    imgFlat = imgBaW.flatten()
    imgFlat.sort()

    # percentage is a number of piwel to saturate atthe low and high ends
    percentage = 50
    # the current max and min values
    minIn = imgFlat[percentage]
    maxIn = imgFlat[imgFlat.shape[0] - percentage]

    G = (imgBaW-minIn)
    scalaire = (maxOut-minOut)/(maxIn - minIn + 0.01)
    G = G*scalaire + minOut
    G = G.astype(int)
    G[imgBaW>maxIn] = 255
    G[imgBaW<minIn] = 0
    return G

def create_circular_mask(h, w, center=None, radius=None):
    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

#This function is useful to treat blacks frames
def circle(im, r):
    (h, l) = im.shape[0], im.shape[1]

    mask = create_circular_mask(h, l, center=None, radius=300)
    mask = np.logical_not(mask)

    img2 = im* mask
    nb = np.count_nonzero(mask)
    s = np.sum(np.sum(img2))

    return s/nb

    
#This function is useful to treat blacks frames
def rayon(im):
  
    L = len(im[0])
    l = len(im)

    rmin = l/4
    rmax = (1/2)*sqrt(l**2 + L**2)
    nb_pas = 10
    pas = (rmax - rmin) / nb_pas
    
    Lrayon = []
    Lmoyenne = []
    
    r = rmin 
    for i in range (nb_pas) : 
        Lrayon.append(r)
        Lmoyenne.append(circle(im,r))
        r += pas
    return [Lrayon, Lmoyenne]
    
#This function is useful to treat blacks frames
def minMoyenne(image):

    Lrayon, Lmoyenne = rayon(image)
    
    Mmin = 256
    indice = -1
    
    for i in range (len(Lrayon)):
        if Lmoyenne[i] < Mmin : 
            Mmin = Lmoyenne[i]
            indice = i     
    return Lrayon[indice - 4]

#This function is useful to treat blacks frames
def masqueBinaire(im):
    rayon = minMoyenne(im)
    
    m,n = im.shape
    m2 = int(np.floor(float(m)/ 2.0))
    n2 = int(np.floor(float(n)/ 2.0))
    mask = copy.copy(im)
        
    x, y = np.meshgrid(range( -n2, n2+np.mod(n,2)), range(-m2,m2+np.mod(m,2)))
    mask = (( np.power(x,2) + np.power(y,2) ) < rayon**2).astype(int)    
    return mask
    
#Otsu implementation
def otsu(im):
    
    mask = copy.copy(im)
    
    mask = masqueBinaire(im)
    
    pxl = im[mask>0]
    pxl_hist,_ = np.histogram(pxl,range(0,256))
    
    totalSigma = np.zeros(256)         
    for k in range (0,256): 
        omega_1 = pxl_hist[0:k].sum().astype(float)
        omega_2 = pxl_hist[k:-1].sum().astype(float)
        mu_1 = np.mean(pxl[pxl<=k])
        if (np.isnan(mu_1)):
            mu_1 = 0
        mu_2 = np.mean(pxl[pxl>k])
        if (np.isnan(mu_2)):
            mu_2 = 0
        totalSigma[k] = omega_1 * omega_2 * ( (mu_1 - mu_2)**2)
        
    seuil = np.argmax(totalSigma)

    mask_out = np.logical_and( mask>0 , im < seuil).astype(int)

    return mask_out

#the connectedComponent function takes a two dimensionnal numpy array corresponding to a segmentation mask
def connectedComponentsRemoval(imgArg):
    #calculation of the connected components using openCV
    img = np.array(imgArg, dtype=np.uint8)

    #connected components data
    (retval, comp, stats, centroid) = cv2.connectedComponentsWithStats(img)

    # we get the connected components areas
    areas = (stats[:, 4])

    maskOut = np.copy(imgArg)
    maskOut = np.array(maskOut, dtype=np.uint8)

    #we sort the areas to get the the 2 maximum areas (the Area Of Iterrest and the rest of the Skin)
    sortedAreas = np.sort(areas)

    #if there is only 2 zones:, the mask is ok
    if (retval <= 2):
        return (imgArg)
    else:
        #for each connected component, if the area is inferior to the second bigge, we fill it with black
        for i in range(0, retval):
            if (areas[i] < sortedAreas[-2]):
                maskOut[comp == i] = 0

    return(maskOut)

def MorphologicalFilling (im):
    kernel = np.ones((45,45), np.uint8)
    closing = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)
    return closing

# This algorithm takes in parameter an image of beauty spot and returns the mask of the hair
def hairMask(im):
    print("check")
    M_colors = []
    
    for color in range(3): # The closing is made on the three channels (red, green, blue) to detect hair
        channel = im[:,:,color] 

        # The closing is done with a rectangular kernel
        SE = cv2.getStructuringElement(cv2.MORPH_RECT,(7,7)) # --> Tester sur : vertical, puis un peu diagonal, puis beaucoup plus, jusqu'à horizontal
        
        # Let's obtain thin, a gray-scale image which show the thin elements of the initial image
        thin = cv2.morphologyEx(channel, cv2.MORPH_CLOSE, SE) - channel
        
        # temp is the matrix of the same size as thin. It is actually  the mask of the hair through the used channel
        temp = np.zeros_like(thin) 
        temp[thin > 15] = 1 
        
        M_colors.append(temp)
    
    # Let's finally obtain the mask, which is the union of the three masks calculated above
    mask = np.logical_or(M_colors[0], M_colors[1], M_colors[2]) 
    
    # Finally, closedMask is the image with less discontinuity
    SE4 = cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))
    closedMask = cv2.dilate(mask, SE4, 2) 
    
    return closedMask

# This algorithm takes in parameter the mask of hairMask and removes the noise. It removes the elements with few neighbours
def hairDetection(mask): 
    
    #Let's find all the connected components
    print("eee")
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    print('aaaa')
    #the following line removes the background of the image. It is actually a component that won't be useful for the rest of the algorithm
    sizes = stats[1:, -1]; nb_components = nb_components - 1
    print('lllllllll')
    
    # min_size is the minimum size of particles that we want to keep (that is to say, the number of pixels)
    min_size = 180
    print('gggggggg')
    
    #img2 will become the mask without noise
    img2 = np.zeros((output.shape))
    print('sssssssssss')

    #for every component in the image, let's keep it only if there are more than min_size elements in it.
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255
    print('scffffff')

    return img2

#takes an image RGB array and a mask corresponding to the hair
#Aims at removing the hair by inpainting them with interpolation
#here : we replace the pixel by the value interpolated from 2 pixels on the sides or above and below
# the distance of the piwels is a parameter
def hairInterpolation1(origin, detect):
    (h,l)= detect.shape

    new = origin.copy()

    #the indices of the hair points
    indices = np.nonzero(detect)

    nbPoints = (indices[0].shape[0])

    #the distance of the 2 points of interpolation
    cSquare = 30
    #going through all the hair points
    for i in range (0, nbPoints):
        x = indices[0][i]
        y = indices[1][i]

        # A and B are the 2 points used to interpolate
        neighbourhood = False
        #a loop to test if the interpolation points are valid (to be improved)
        while  (not neighbourhood):
            xA,yA = min (int(x /cSquare)*cSquare,  h-1), y
            xB,yB = min(xA + cSquare, h-1), y

            if (  detect[xA][yA]  or detect[xB][yB] ):
                xA, yA = x, min ( int( (y /cSquare)*cSquare), l-1 )
                xB, yB = x, min((yA + cSquare), l-1)
            neighbourhood = True

        if all(x >= 0 for x in (xA, yA, xB, yB)) and all(x < h for x in (xA, xB)) and all(x < l for x in (yA, yB)):
            for k in range (0,3):
                #we interpolate
                u = ((x-xA + y-yA)/float(cSquare))
                new[x][y][k] = int(origin[xA][yA][k]*u + (1-u)*origin[xB][yB] [k]  )

    #plt.show()
    return (new)

#For each hair mask pixel, we calculate the direction of the belonging hair, using linear regression
#and replace the pixel by interpolating it with two pixels chosen on a perpendicular direction
def hairInterpolationNormalMethod(origin, detect):
    (h, l) = detect.shape
    #detect = np.logical_not(detect)

    new = origin.copy()
    indices = np.nonzero(detect)
    nbPoints = (indices[0].shape[0])
    #size of the zone where the linreg is computed
    zoneSize = 5
    #half of the distance between the interpolation pixels
    interpolationMaxSize = 20
    print(nbPoints)

    #we take the skeleton of the  mask in order to calculate the directions of the hair
    skeleton = skimage.morphology.skeletonize(detect)

    for i in range (0,int(nbPoints/1)):
        x = indices[0][i]
        y = indices[1][i]
        #print (str(i) + "/" + str(nbPoints))

        #limits of the zone of linear regression
        left_size = min (zoneSize, y)
        right_size = min (zoneSize, l-y)
        up_size = min(zoneSize, x)
        down_size = min(zoneSize, h-x)
        subArraySkeleton = skeleton[x-up_size:x+down_size, y-left_size:y+right_size]
        subBones = np.nonzero(subArraySkeleton)
        nBones = subBones[0].shape[0]

        if (nBones != 0):
            #linear regression
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(subBones[0], subBones[1])
            if (  not math.isnan(slope) ):
                normalX, normalY = (slope/float(np.sqrt(slope**2 + 1))), (-1/(float(np.sqrt(slope**2 + 1))))

                xA = (x + int(normalX * interpolationMaxSize))
                yA = y + int(normalY * interpolationMaxSize)

                xB = (x - int(normalX * interpolationMaxSize))
                yB = (y - int(normalY * interpolationMaxSize) )

                #we interpolate linearly
                if all(x >= 0 for x in (xA, yA,xB,yB) ) and all(x < h for x in (xA, xB)) and all(x < l for x in (yA, yB)) :
                    u = (np.sqrt((xA-x)**2 + (yA-y)**2))/float(np.sqrt(  (xA-xB)**2 + (yA-yB)**2) + 0.01 )
                    for k in range(0, 3):
                        new[x][y][k] = int(new[xA][yA][k] * u + (1 - u) * new[xB][yB][k])

            #else:
                #print("no slope")
        #else :
            #print("no skeleton around the point")

    return(new)

def main(src):
    print("a")
    img = cv2.imread(src)
    print("b")
    imMask = hairMask(img)
    print("c")
    imDetection = hairDetection(imMask)
    print("d")
    imInterpolation = hairInterpolation1(img, imDetection)
    print("e")
    imgBaW = ColorSpaceTransformation(imInterpolation)
    print("f")
    imNoNoise = noiseRemoval(imgBaW)
    print("g")
    imContrast = intensityAdjust(imNoNoise)
    print("h")
    imgBinaire = otsu(imContrast)
    print("i")
    imgFiltree = connectedComponentsRemoval(imgBinaire)
    print("j")
    imgFinale = MorphologicalFilling(imgFiltree)
    print("k")
    return imgFinale

##
class Interface(Frame):
    
    
    def __init__(self, fenetre, **kwargs):
        
        Frame.__init__(self, fenetre, width=GetSystemMetrics(0), height=GetSystemMetrics(1), relief='ridge', **kwargs)    #création d'une interface qui hérite de la classe Frame de tkinter
        self.pack(fill=BOTH)

        self.canva=Canvas(self,width=0.86*self.winfo_screenwidth(),height=0.911*self.winfo_screenheight())   #création d'une zone de dessin où afficher l'image
        self.canva.grid(row=0,column=0,columnspan=2000,rowspan=500)
        
        bookman20=Font(family='Bookman', size=13, weight=BOLD) #police personnalisée pour les boutons

        self.bouton_ouvrir = Button(self, text="Ouvrir", bg="white smoke", height=1, command=self.charger_image, font=bookman20)  #création du bouton ouvrir (une image)        
        self.bouton_aide=Button(self, text="Aide", bg="white smoke", height=1,command=self.aide, font=bookman20) #création du bouton d'aide 
        self.bouton_aide.grid(row=0, column=4)
        self.bouton_ouvrir.grid(row=0,column=0)

        self.bouton_segmentation = Button(self, text="Segmentation",fg="black", height=1 , bg="OliveDrab2", command=self.segmenter, font=bookman20)#création du bouton segmentation   
        self.bouton_segmentation.grid(row=0, column=1)
        
        self.bouton_extraction = Button(self, text="Extraction & Classification",fg="black" , height=1,bg="coral1", command=self.extraire_caracs, font=bookman20)#création du bouton extraction&classification  
        self.bouton_extraction.grid(row=0,column=2)
    
        self.bouton_sauvegarder = Button(self, text="Sauvegarde",bg="white smoke", height=1, fg='black', command=self.sauvegarder, font=bookman20)#création du bouton de sauvegarde 
        self.bouton_sauvegarder.grid(row=0,column=3)
     
    #appuyer sur le bouton aide produit une fenêtre d'information   
    def aide(self):
        showinfo(title="Aide", message="Le mélanome est le cancer de la peau le plus mortel. D'après l'OMS, il est la cause d'environ 60000 décès par an dans le monde. Le dépistage précoce du mélanome est important pour traiter la maladie au plus vite et retirer la tumeur par une simple excision.\n\nOuvrir : Permet de charger et d'afficher une image dont la taille convient à celle de l'écran.\n\nSegmentation : A partir de l'image ouverte, un masque binaire qui délimite le grain de beauté est créé. Pour se faire, on se restreint d'abord au canal bleu de l'image. On supprime ensuite les poils apparents. La détection du grain de beauté se fait par seuillage via la méthode d'Otsu. Enfin, les composantes résiduelles sont retirées.\n\nExtraction&Classification : L’extraction de features consiste à déterminer différentes caractéristiques des grains de beauté étudiés grâce à des méthodes codées en Python. Ce processus suit la méthode ABCD (asymétrie, bordure, couleur et diamètre).\nLes données obtenues sont par la suite utilisées dans des algorithmes de Machine Learning préalablement entraînés pour déclarer si le grain de beauté en question est bénin ou malin.", default=OK, icon=INFO)
    #la fonction qui s'execute lors de l'appui sur le bouton "Ouvrir"
    def charger_image(self):
        global im, im_seg,results              
        im,im_seg,results=None,None,None        #cette ligne et la suivante permette de réinitialiser les variables images, image_segmentée et les résultats à l'ouverture d'une image
        del im,im_seg,results                     
        self.canva.delete("all")
        global filepath   #on déclare filepath global pour qu'elle soit accessible dans les autres fonctions de boutons
        filepath = askopenfilename(filetypes=[("Image Files","*.jpg;*.png")])
        im=cv2.imread(filepath)
        im=cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  #opencv ouvre des image en BGR. On la convertit en RGB
        he,wi=im.shape[:2]
        global wsize,baseheight
        baseheight = self.winfo_height()
        hpercent = (baseheight / float(he))
        wsize = int((float(wi) * float(hpercent)))
        im_redimensionnee = cv2.resize(im,(wsize, baseheight)) #on redimensionne l'image de sorte que sa hauteur soit celle de la fenêtre principale
        #self.canva['width']= wsize+ (self.winfo_screenwidth()-wsize)/4.25
        global photo
        photo= PIL.ImageTk.PhotoImage(PIL.Image.fromarray(im_redimensionnee))
        fenetre.geometry("%dx%d+0+0" % (wsize, baseheight)) #on redimensionne la fenêtre à la taille de l'image
        self.canva.create_image(0,0,image =photo, anchor='nw') #on affiche l'image sur le canva
        
    #la fonction qui s'execute lors de l'appui sur le bouton "Segmentation"
    def segmenter(self):
        
        try:
            global im_seg #on déclare im_seg global pour qu'elle soit accessible dans les autres fonctions de boutons
            im_seg=main(filepath) #on execute la fonction main de la semgentation qui retourne le masque binaire
            #im_seg=im
            im_seg=im_seg.astype('uint8')
            detectedContours, hierarchy = cv2.findContours(im_seg,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE) #à partir du masque binaire, on dessine les contouts du grain de beauté sur l'image initiale
            for contour in detectedContours :
                cv2.drawContours(im,contour, -1, (0,0,255), 3)
            im_redimensionnee = cv2.resize(im,(wsize, baseheight))
            self.canva['width']= wsize+ (self.winfo_screenwidth()-wsize)/4.25
            global photo
            photo= PIL.ImageTk.PhotoImage(PIL.Image.fromarray(im_redimensionnee))
            self.canva.delete("all")
            self.canva.create_image(0,0,image =photo, anchor='nw') #on affiche l'image segmentée redimensionnée
            
        except NameError:
            im_exists = 'filepath' in locals() or 'filepath' in globals()
            if not im_exists:   
                showerror(title="Erreur", message="Vous n'avez pas ouvert d'image.", default=OK, icon=ERROR) #ceci est un message de précaution si aucune image n'a encore été ouverte et qu'on appuie sur le bouton de segmentation


    #la fonction qui s'execute lors de l'appui sur le bouton "Extraction&Classification"
    def extraire_caracs(self):
        
        try:
            m=im     #Si im ou im_seg n'est pas définie (on n'a pas encore ouvert d'image ou pas encore segmentée l'image ouverte, on aura un NameError d'où un message de précaution de notre part
            n=im_seg
            scores = Tk()           #on va afficher une nouvelle fenêtre qui contiendra les résultats.
            scores.title('Extraction des caractéristiques')

            style = Style(scores)

            style.configure('Treeview', rowheight=40)                 #Permet de changer la taille du tableau qu'on va créer, ainsi que la police d'écriture
            style.configure('Treeview', font=(None, 15))
            style.configure('Treeview.Heading', font=(None, 18, 'bold'))                
            cols = ('Caractéristiques', 'Scores')
            listBox = Treeview(scores, height=7, columns=cols, show='headings') #Le tableau est un Treeview
            for col in cols:
                listBox.heading(col, text=col)       #on regle les titres des colonnes du tableau et leur taille
                listBox.column(col, width=int(1/6*GetSystemMetrics(0)))

            listBox.grid(row=0, column=0, columnspan=2)

            L=ABCDbis(filepath,255*im_seg)   #les  résultats de l'extraction des caractéristiques
            A1,A2,B1,B2,C=L[0]
            
            global results
            results = [['',''],['A1 - Assymetry 1',str(A1)], ['A2 - Assymetry 2',str(A2)],['B1 - Borders 1',str(B1)], ['B2 - Borders 2',str(B2)], ['C - Colors' , str(C)]] 
            listDiag=diagnostic(filepath,255*im_seg)
            print(listDiag)
            lis=[]
            for x in listDiag:
                if x:
                    lis.append('Malin')
                else:
                    lis.append('Bénin')
            for i in range(len(results)):
                listBox.insert("", "end", values=(results[i][0], results[i][1]))  #on insère les caractéristiques extraites dans le tableau
                
            Label(scores, text="Linear Discriminant Analysis", font=("Helvetica 16 bold"), justify=LEFT).grid(row=1, columnspan=1,sticky=W)  #on donne les résultats de classification selon différentes méthodes
            Label(scores, text=lis[0], font=("Helvetica 16"), justify=LEFT).grid(row=1, column=1,sticky=W)
            Label(scores, text="Quadratic Discriminant Analysis", font=("Helvetica 16 bold"), justify=LEFT).grid(row=2, columnspan=1,sticky=W)
            Label(scores, text=lis[1], font=("Helvetica 16"), justify=LEFT).grid(row=2, column=1,sticky=W)
            Label(scores, text="Naive Bayes", font=("Helvetica 16 bold"),justify=LEFT).grid(row=3, columnspan=1,sticky=W)
            Label(scores, text=lis[2], font=("Helvetica 16"), justify=LEFT).grid(row=3, column=1,sticky=W)
            #Label(scores, text="K-NearestNeighbours", font=("Helvetica 16 bold"), justify=LEFT).grid(row=4, columnspan=1,sticky=W)
            #Label(scores, text="Malin", font=("Helvetica 16"), justify=LEFT).grid(row=4, column=1,sticky=W)
            Label(scores, text="Logistic Regression", font=("Helvetica 16 bold"), justify=LEFT).grid(row=4, columnspan=1,sticky=W)
            Label(scores, text=lis[3], font=("Helvetica 16"), justify=LEFT).grid(row=4, column=1,sticky=W)

            scores.mainloop()
            scores.destroy()

          
        except NameError:
            im_exists = 'im' in locals() or 'im' in globals()
            im_seg_exists = 'im_seg' in locals() or 'im_seg' in globals()
            if not im_exists:  #ce sont des messages de précaution si aucune image n'a encore été ouverte et qu'on appuie sur le bouton de segmentation ou si on veut extraire les caracs et que la segmentation n'a pas encore été faite
                showerror(title="Erreur", message="Vous n'avez pas ouvert d'image.", default=OK, icon=ERROR)
            elif not im_seg_exists:
                showerror(title="Erreur", message="Vous n'avez pas encore effectué la segmentation.", default=OK, icon=ERROR)

    #la fonction qui s'execute lorsqu'on clique sur le bouton 'Confirmer' de la fenêtre de sauvegarde            
    def save_quit_infos(self):
        try:
            a=direct #si le chemin de sauvegarde n'est pas défini, on a un NameError
            self.last_name=self.e1.get() #on récupère ce qui a été écris dans les champs de texte
            self.first_name=self.e2.get()
            self.age=self.e3.get()
            self.date=self.e4.get()
            self.quit()
        except NameError:
            showerror(title="Erreur", message="Vous n'avez pas défini de lieu de sauvegarde", default=OK, icon=ERROR)
            self.infos.update() #permet de mettre la fenêtre de sauvegarde au premier plan
            self.infos.deiconify()
            
    #la fonction qui s'execute lorsqu'on clique sur le bouton 'Sauvegarder sous' de la fenêtre de sauvegarde            
    def save_as(self):
        global direct
        direct=askdirectory() #on récupère le chemin choisi
        self.infos.textvar.set(str(direct)) #on affiche ce chemin dans le label correspondant
        self.infos.update()   #On remet la fenêtre de sauvegarde au premier plan
        self.infos.deiconify()
        
    #la fonction qui s'execute lorsqu'on clique sur le bouton 'Sauvegarder' de la fenêtre principale              
    def sauvegarder(self):
        try:
            m=im #si une des étapes n'a pas encore été faite, un NameError apparaîtra.
            n=im_seg
            l=results
            self.infos=Tk() #on crée la fenêtre de sauvegarde des informations du patient
            self.infos.title('Informations du patient')
            Label(self.infos, text="Nom", font='Helvetica 18', anchor='w').grid(row=0, sticky=W)
            Label(self.infos, text="Prénom", font='Helvetica 18', anchor='w').grid(row=1, sticky=W)
            Label(self.infos, text="Âge", font='Helvetica 18', anchor='w').grid(row=2, sticky=W)
            Label(self.infos, text="Date", font='Helvetica 18', anchor='w').grid(row=3, sticky=W)
            Label(self.infos, text="    ").grid(row=0,column=1)
            Label(self.infos, text="    ").grid(row=1,column=1)
            Label(self.infos, text="    ").grid(row=2,column=1)
            Label(self.infos, text="    ").grid(row=3,column=1)

            self.e1 = Entry(self.infos, font='Helvetica 18', width=20) #on crée les champs d'entrée de chaque information (Nom, Prénom, Age, ..)
            self.e2 = Entry(self.infos, font='Helvetica 18', width=20)
            self.e3 = Entry(self.infos, font='Helvetica 18', width=20)
            self.e4 = Entry(self.infos, font='Helvetica 18', width=20)
            self.e1.grid(row=0, column=2)
            self.e2.grid(row=1, column=2)
            self.e3.grid(row=2, column=2)
            self.e4.grid(row=3, column=2)
            self.bouton_confirmer=Button(self.infos, text='Confirmer', font='Helvetica 12', command=self.save_quit_infos) #on crée les boutons sauvegarder_sous et confirmer de la fenêtre de sauvegarde
            self.bouton_sauvegarder_sous=Button(self.infos, text='Sauvegarder sous', font='Helvetica 12',command=self.save_as)
            self.bouton_confirmer.grid(row=5, column=0, columnspan=1)
            self.bouton_sauvegarder_sous.grid(row=4, column=0, columnspan=1)
            self.infos.textvar=StringVar(master=self.infos)
            self.savelabel=Label(self.infos,textvariable=self.infos.textvar, width=30, font='Helvetica 12').grid(row=4,column=1, columnspan=2)
            self.infos.mainloop()
            self.infos.destroy()
            
            os.mkdir(direct+'/'+self.last_name)     #On crée un dossier dans le lieu de sauvegarde, au nom du nom de famille
            copyfile(filepath,direct+'/'+self.last_name+'/'+self.last_name+'.png')
            os.chdir(direct+'/'+self.last_name)
            cv2.imwrite(self.last_name+'_seg.png', cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
           
            cv2.imwrite(self.last_name+'_mask.png', cv2.cvtColor(255*im_seg, cv2.COLOR_GRAY2BGR))
            f=open(self.last_name+'.txt','w+')
            f.write('********** Résultats **********\n\n')
            f.write('Nom : '+self.last_name+'\nPrénom : '+self.first_name+'\nÂge : '+self.age+' ans\nDate : '+self.date+'\n\n')
            for r in results:
                f.write(r[0]+' : '+r[1]+'\n')
            f.write('\nLinear Discriminant Analysis : Bénin\nQuadratic Discriminant Analysis : Bénin\nNaive Bayes: Bénin\nK-NearestNeighbours : Malin\nLogistic Regression : Bénin')
            f.close()
        except NameError:
            im_exists = 'im' in locals() or 'im' in globals()
            im_seg_exists = 'im_seg' in locals() or 'im_seg' in globals()
            im_extr_exists = 'results' in locals() or 'results' in globals()
            if not im_exists:
                showerror(title="Erreur", message="Vous n'avez pas ouvert d'image.", default=OK, icon=ERROR)
            elif not im_seg_exists:
                showerror(title="Erreur", message="Vous n'avez pas encore effectué la segmentation.", default=OK, icon=ERROR)
            elif not im_extr_exists:
                showerror(title="Erreur", message="Vous n'avez pas encore extrait les caractéristiques de l'image.", default=OK, icon=ERROR)

       

fenetre = Tk()
fenetre.title("Détecteur de mélanomes")
w, h = 0.75*fenetre.winfo_screenwidth(), fenetre.winfo_screenheight()
fenetre.geometry("%dx%d+0+0" % (w, h))

interface = Interface(fenetre)
interface.mainloop()
interface.destroy()
