import numpy as np
import cv2 as cv
import glob
import matplotlib.pyplot as plt

img1 = cv.imread('tests/+2blue.jpg')
sift = cv.SIFT_create()# queryImage
kp1, des1 = sift.detectAndCompute(img1, None)
path = glob.glob("images/+*.jpg")
cv_img = []
l=0

for img in path:
    img2 = cv.imread(img) # trainImage
    # Initiate SIFT detector

    # find the keypoints and descriptors with SIFT

    kp2, des2 = sift.detectAndCompute(img2,None)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    # Need to draw only good matches, so create a mask

    # ratio test as per Lowe's paper

    if (l < len(matches)):
        l = len(matches)
        image = img2
        match = matches

matchesMask = [[0,0] for i in range(len(match))]
for i,(m,n) in enumerate(match):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]
draw_params = dict(matchColor = (0,255,0),
                    singlePointColor = (255,0,0),
                    matchesMask = matchesMask,
                    flags = cv.DrawMatchesFlags_DEFAULT)

img3 = cv.drawMatchesKnn(img1,kp1,image,kp2,match,None,**draw_params)

plt.imshow(img3),plt.show()