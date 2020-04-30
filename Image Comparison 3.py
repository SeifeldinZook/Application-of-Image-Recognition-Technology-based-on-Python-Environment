import cv2
import numpy as np

original = cv2.imread('/Users/zook/Pycharm Projects/Images/Image Comparison/original.jpg')
compared = cv2.imread('/Users/zook/Pycharm Projects/Images/Image Comparison/original.jpg')  # Note: these images are read as arrays

# ---------------------------------------------------------------------------------------------------------------
''' 1) Check if the images are equal. '''
# ---------------------------------------------------------------------------------------------------------------

if original.shape == compared.shape:
    print('The Images have same dimensions and channels.')
    difference = cv2.subtract(original, compared)
    r, g, b = cv2.split(difference)
    if cv2.countNonZero(r) == cv2.countNonZero(g) == cv2.countNonZero(b) == 0:
        print("And The Channels have same pixels'value!")
    else:
        print("But The Channels have different pixels'value!")
else:
    print('The Images have different dimensions or channels')
    print('Original(X,Y,RGB): ', original.shape)
    print('Compared(X,Y,RGB): ', compared.shape)

# ---------------------------------------------------------------------------------------------------------------
''' 2) Check for similarities between 2 Images. '''
''' 3) Detect how similar two images are?. '''
# ---------------------------------------------------------------------------------------------------------------

sift = cv2.xfeatures2d.SIFT_create()
kp_1, desc_1 = sift.detectAndCompute(original, None)
kp_2, desc_2 = sift.detectAndCompute(compared, None)
print('No. of Keypoints of Original is: ', str(len(kp_1)))
print('No. of Keypoints of Compared is: ', str(len(kp_2)))

index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(desc_1, desc_2, k=2)

good_points = []
ratio = 0.6
for m, n in matches:
    if m.distance < ratio * n.distance:
        good_points.append(m)
print('Good points= ', len(good_points))

result = cv2.drawMatches(original, kp_1, compared, kp_2, good_points, None)

Lower_keypoints = 0            # Define how similar they are
if len(kp_1) <= len(kp_2):
    Lower_keypoints = len(kp_1)
else:
    Lower_keypoints = len(kp_2)

print('Percentage of how similar two images are: ', len(good_points) / Lower_keypoints * 100)

cv2.imshow("Original", cv2.resize(original, None, fx=0.3, fy=0.3))
cv2.imshow("Compared", cv2.resize(compared, None, fx=0.3, fy=0.3))
cv2.imshow("result", cv2.resize(result, None, fx=0.3, fy=0.3))
cv2.imwrite("feature_matching.jpg", result)
cv2.waitKey(0)
cv2.destroyAllWindows()