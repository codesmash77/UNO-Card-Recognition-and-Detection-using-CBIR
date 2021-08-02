import numpy as np
import cv2 as cv
import glob
import matplotlib.pyplot as plt

img1 = cv.imread('tests/1blue.jpg')
orb = cv.ORB_create()# queryImage
kp1, des1 = orb.detectAndCompute(img1, None)
path = glob.glob("images/*.jpg")
cv_img = []
l=0


for img in path:
    img2 = cv.imread(img) # trainImage
    # Initiate SIFT detector

    # find the keypoints and descriptors with SIFT

    kp2, des2 = orb.detectAndCompute(img2, None)
    # feature matching
    bf = cv.BFMatcher(cv.NORM_L1, crossCheck=True)

    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    if(l<len(matches)):
        l=len(matches)
        image=img2
        m=matches


kp2, des2 = orb.detectAndCompute(image,None)
img3 = cv.drawMatches(img1, kp1, image, kp2, m, image, flags=2)
plt.imshow(img3), plt.show()

