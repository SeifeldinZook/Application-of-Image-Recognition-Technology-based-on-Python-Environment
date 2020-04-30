import cv2
import numpy as np

original = cv2.imread('/Users/zook/Pycharm Projects/Images/Thesis/1.2.PNG')
compared = cv2.imread('/Users/zook/Pycharm Projects/Images/Thesis/1.3.PNG')
# Note: these images are read as arrays

# ----------------------------------------------------------------------------------------------------
''' 1) Check if the images are equal. '''
# ----------------------------------------------------------------------------------------------------
or_shape = original.shape   # (y-axis, x-axis, channels) (1079, 1920, 3)
co_shape = compared.shape   # Note: .shape is an array function
if or_shape == co_shape:
    print('The Images have same dimensions and channels.')
    difference = cv2.subtract(original, compared)
    r, g, b = cv2.split(difference)
    if cv2.countNonZero(r) == cv2.countNonZero(g) == cv2.countNonZero(b) == 0:
        print("And The Channels have same pixels'value!")
    else:
        print("But The Channels have different pixels'value!")
else:
    print('The Images have different dimensions or channels')
    print('Original(X,Y,RGB): ', or_shape)
    print('Compared(X,Y,RGB): ', co_shape)

# ----------------------------------------------------------------------------------------------------
# ''' 2) Check for similarities between 2 Images. '''
# ----------------------------------------------------------------------------------------------------
sift = cv2.xfeatures2d.SIFT_create()     # load the sift algorithm
or_Keypoints = sift.detect(original, None)
co_Keypoints = sift.detect(compared, None)
'''sift.detect() function finds the keypoint in the images. 
Each keypoint is a special structure which has many attributes like its (x,y) coordinates, 
size of the meaningful neighbourhood, angle which specifies its orientation, 
response that specifies strength of keypoints etc.'''

or_Keypoint_DRAW = cv2.drawKeypoints(original, or_Keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# cv2.imshow('Original Keypoints', or_Keypoint_DRAW)
co_Keypoint_DRAW = cv2.drawKeypoints(compared, co_Keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
co_Keypoint_DRAW = cv2.drawKeypoints(compared, co_Keypoints, None)
'''cv2.drawKeyPoints() function which draws the small circles on the locations of keypoints.
If you pass a flag, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS to it,
it will draw a circle with size of keypoint and it will even show its orientation. '''
# cv2.imshow('Compared Keypoints', cv2.resize(co_Keypoint_DRAW, None, fx=0.4, fy=0.4))

'''Since you already found keypoints,
you can call sift.compute() which computes the descriptors from the keypoints we have found.
Eg: kp,des = sift.compute(gray,kp)
If you didn't find keypoints, directly find keypoints and descriptors in a single step with the function,
sift.detectAndCompute(). '''

kp_1, desc_1 = sift.detectAndCompute(original, None)
# kp_1 will be a list of keypoints and desc_1 is a numpy array of shape (No.of Keypoints*128)
kp_2, desc_2 = sift.detectAndCompute(compared, None)

index_params = dict(algorithm=0, trees=5)   # Load FlannBasedMatcher....
search_params = dict()                      # ....which it the method used to find the matches between
flann = cv2.FlannBasedMatcher(index_params, search_params)    # ....the descriptors of the 2 images.
'''FLANN based Matcher. FLANN stands for Fast Library for Approximate Nearest Neighbors. 
It contains a collection of algorithms optimized for fast nearest neighbor search
in large datasets and for high dimensional features.
It works more faster than BFMatcher for large datasets.
For FLANN based matcher, we need to pass two dictionaries which specifies the algorithm to be used, 
its related parameters etc. First one is IndexParams.
SearchParams. It specifies the number of times the trees in the index should be recursively traversed.
Higher values gives better precision, but also takes more time.
If you want to change the value, pass search_params = dict(checks=100).'''

matches = flann.knnMatch(desc_1, desc_2, k=2)
# find the matches between the 2 images. We’re storing the matches in the array ‘matches’.
# The array will contain all possible matches, so many false matches as well.

good_points = []   # Apply the ratio test to select only the good matches.
ratio = 0.6
for m, n in matches:
    if m.distance < ratio * n.distance:
        good_points.append(m)
print('Good points= ', len(good_points))
'''The quality of a match is define by the distance.
The distance is a number, and the lower this number is, the more similar the features are.
By applying the ratio test we can decide to take only the matches with lower distance, so higher quality.
If you decrease the ratio value, for example to 0.1 you will get really high quality matches,
but the downside is that you will get only few matches.
If you increase it you will get more matches but sometimes many false ones.'''

result = cv2.drawMatches(original, kp_1, compared, kp_2, good_points, None)

cv2.imshow("Original", original)
'''If you want to resize the image show: 
cv2.imshow("Original", cv2.resize(original, None, fx=0.3, fy=0.3))'''

cv2.imshow("Compared", compared)
cv2.imshow("result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
