# UNO-Card-Recognition-and-Detection-using-CBIR
 
For this project, two approaches to detecting and identifying UNO cards were
explored.

Approach 1: Using sift algorithm to find keypoints and descriptors and flann’s
algorithm to show the matches.This approach is designed around the Scale Invariant Feature Transform [1](SIFT) algorithm. The idea is to find the best
match between the SIFT descriptors in the target image with SIFT descriptors
that have been computed for the training images using Flann’s algorithm.

Approach 2: Using ORB algorithm to find keypoints and descriptors and
brute force algorithm to show the matches.This approach is designed around
the Oriented FAST and Rotated BRIEF [11](ORB) algorithm. The idea is to
find the best match between the ORB descriptors in the target image with ORB
descriptors that have been computed for the training images using Brute Force
algorithm.