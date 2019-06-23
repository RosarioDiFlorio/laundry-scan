import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import sys

scene_path = sys.argv[1]
template_path = sys.argv[2]


##DOCS
'''
https://docs.opencv.org/3.4/d5/d3c/classcv_1_1xfeatures2d_1_1SIFT.html
https://docs.opencv.org/3.1.0/da/df5/tutorial_py_sift_intro.html
https://docs.opencv.org/3.1.0/dc/de2/classcv_1_1FlannBasedMatcher.html
https://docs.opencv.org/2.4/modules/flann/doc/flann_fast_approximate_nearest_neighbor_search.html
'''

'''
int 	nfeatures = 0,
int 	nOctaveLayers = 3,
double 	contrastThreshold = 0.04,
double 	edgeThreshold = 10,
double 	sigma = 1.6 
'''
##template sites
## http://www.textileaffairs.com/hirez.htm

##le size dei template saranno sempre le stesse.
##simboli diversi hanno design diversi (naggia a maronn ) <- questo é il problema piú grande mannaggia a mamma re designer
##bisogna trainare sulle foto.
###sembra avere importanza anche la size dell'immagine
img1 = cv.imread(template_path,cv.IMREAD_GRAYSCALE)          # queryImage
img1 = cv.resize(img1, (0,0), fx=2, fy=2) 

img2 = cv.imread(scene_path,cv.IMREAD_GRAYSCALE) # trainImage
img2 = cv.resize(img2, (0,0), fx=1, fy=1) 
# Initiate SIFT detector
#IRON 2 dot and cross sigma 3 seems good.
#SiGMA is important to take the point on the surface, threshold depends how many features we should consider, layer is slowing but better precision
sift = cv.xfeatures2d.SIFT_create( nfeatures=0, nOctaveLayers=3, contrastThreshold=0.03, edgeThreshold=10, sigma=3)
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)

sift2 = cv.xfeatures2d.SIFT_create( nfeatures=0, nOctaveLayers=3, contrastThreshold=0.02, edgeThreshold=5, sigma=6)
kp2, des2 = sift2.detectAndCompute(img2,None)
# FLANN parameters
FLANN_INDEX_KDTREE = 2
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=100)   # or pass empty dictionary
flann = cv.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2,compactResult = True)
# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]



##

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.8 *n.distance:
        matchesMask[i]=[1,0]
draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = cv.DrawMatchesFlags_DEFAULT)
print(matchesMask)
img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
plt.imshow(img3,),plt.show()