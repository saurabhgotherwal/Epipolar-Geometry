import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.cm as cm
UBIT = "sgotherw"
np.random.seed(sum([ord(c) for c in UBIT]))

FOLDER_PATH = ''

def ReadImage(image,color=1):
    #read colored image
    img_color = cv2.imread(FOLDER_PATH + image,color)
    return img_color

def WriteImage(imageName, image):    
    cv2.imwrite(FOLDER_PATH + imageName,image)
    return

def ConvertToGrayScale(image):
    img_gray= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    return img_gray

#Detect SIFT keypoints
def FindSIFTKeypointsAndDescriptors(image,img_gray, outputImageName):      
    
    sift = cv2.xfeatures2d.SIFT_create()
    #finds the keypoint in the image    
    keyPoints,descriptors = sift.detectAndCompute(img_gray,None)
    img_output = cv2.drawKeypoints(image,keyPoints,None)
    WriteImage(outputImageName,img_output)
    return keyPoints,descriptors

def MatchKeyPoints(descriptors1,descriptors2, k = 2):    
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1,descriptors2, k=k)      
    return matches

def GetGoodMatches(descriptors1,descriptors2, k = 2):    
    matches = MatchKeyPoints(descriptors1,descriptors2, k)
    # filter good matches
    goodMatches = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            goodMatches.append(m)            
    return goodMatches

def FindFundamentalMatrix(goodMatches):
    sourcePoints = np.int32([ keyPoints1[m.queryIdx].pt for m in goodMatches ])
    destinationPoints = np.int32([ keyPoints2[m.trainIdx].pt for m in goodMatches ])
    fundamentalMatrix, mask = cv2.findFundamentalMat(sourcePoints, destinationPoints, cv2.FM_RANSAC)
    print('Fundamental Matrix:')
    print(fundamentalMatrix)
    return fundamentalMatrix, mask, sourcePoints,destinationPoints

def Drawlines(img1,img2,lines,pts1,pts2):
    colors = [(255,90,0), (10,255,0),(0,0,255),(255,255,0),(255,0,255),(155,205,125),(155,256,0),(255,0,150),(95,85,225),(155,60,90)]
    iteration = 0
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = colors[iteration]        
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
        iteration += 1
    return img1

def GetEpilines(pts,fundamentalMatrix,value):
    lines = []
    lines = cv2.computeCorrespondEpilines(pts.reshape(-1,1,2), value,fundamentalMatrix)
    lines = lines.reshape(-1,3)
    return lines

def GetRandomInliers(points1Inliers,points2Inliers,mask,numberOfSamples): 
    inliersIndexes = np.where(np.asarray(mask.ravel()) == 1)[0]
    randomInliersIndexes = np.random.choice(inliersIndexes,numberOfSamples)    
    randomPoints1Inliers = [points1Inliers[i] for i in randomInliersIndexes]
    randomPoints2Inliers = [points2Inliers[i] for i in randomInliersIndexes]
    return np.asarray(randomPoints1Inliers),np.asarray(randomPoints2Inliers)

def GetDisparity(image1,image2):
    stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
    disparity = stereo.compute(image1,image2)
    return disparity


img1_color = ReadImage('tsucuba_left.png')
img2_color = ReadImage('tsucuba_right.png')

img1_gray = ConvertToGrayScale(img1_color) 
img2_gray = ConvertToGrayScale(img2_color)

img1 = ReadImage('tsucuba_left.png',0) 
img2 = ReadImage('tsucuba_right.png',0)

# find the keypoints and descriptors with SIFT
keyPoints1, descriptors1 = FindSIFTKeypointsAndDescriptors(img1_color,img1,'task2_sift1.jpg')
keyPoints2, descriptors2 = FindSIFTKeypointsAndDescriptors(img2_color,img2,'task2_sift2.jpg')

goodMatches = []
goodMatches = GetGoodMatches(descriptors1,descriptors2, k=2)

img_drawMatches = cv2.drawMatches(img1_color,keyPoints1,img2_color,keyPoints2,goodMatches,None,flags = 0)

WriteImage('task2_matches_knn.jpg',img_drawMatches)

fundamentalMatrix, mask, sourcePoints,destinationPoints = FindFundamentalMatrix(goodMatches)

randomPoints1Inliers, randomPoints2Inliers = GetRandomInliers(sourcePoints,destinationPoints, mask,10)

epilines1 = []
epilines2 = []
epilines1 = GetEpilines(randomPoints2Inliers,fundamentalMatrix,2) 
img_epiline1 = Drawlines(img1,img2,epilines1,randomPoints1Inliers,randomPoints2Inliers)

epilines2 = GetEpilines(randomPoints1Inliers,fundamentalMatrix,1)
img_epiline2 = Drawlines(img2,img1,epilines2,randomPoints2Inliers,randomPoints1Inliers)

WriteImage('task2_epi_right.jpg',img_epiline1)
WriteImage('task2_epi_left.jpg',img_epiline2)

img1_color = ReadImage('tsucuba_left.png',0)
img2_color = ReadImage('tsucuba_right.png',0)

disparity = GetDisparity(img1_color,img2_color)

plt.imsave(FOLDER_PATH + 'task2_disparity.jpg',disparity,cmap=cm.gray,format="png")
