import cv2
import numpy as np
import glob

# ---------------------------------------------------------------------------------------------------------------
''' 4) Check if a set of images match the original one'''
# ---------------------------------------------------------------------------------------------------------------

original = cv2.imread('/Users/zook/Pycharm Projects/Images/Image Comparison/original.jpg')

sift = cv2.xfeatures2d.SIFT_create()
kp_1, desc_1 = sift.detectAndCompute(original, None)

index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Load all the images
all_images_to_compare = []  # create an empty array where we’re going to load later all the images.
titles = []                 # create an empty array where we’re going to load the titles of the images.
for ImPath in glob.iglob(r"images/Image Comparison/*"):  # loop trough all the images inside the folder “images”.
    image = cv2.imread(ImPath)   # load each image and then we add title and image to the arrays we created before.
    titles.append(ImPath)
    all_images_to_compare.append(image)
    # print(ImPath)                      # f will be all the images path

for compared, titles in zip(all_images_to_compare, titles):
    # 1) Check if 2 images are equals
    if original.shape == compared.shape:
        print("----Title: " + titles + "----")
        print("The images have same size and channels")
        difference = cv2.subtract(original, compared)
        r, g, b = cv2.split(difference)
        if cv2.countNonZero(r) == cv2.countNonZero(g) == cv2.countNonZero(b) == 0:
            print("And The Channels have same pixels'value!")
        else:
            print("But The Channels have different pixels'value!")

        # 2) Check for similarities between the 2 images
        kp_2, desc_2 = sift.detectAndCompute(compared, None)

        matches = flann.knnMatch(desc_1, desc_2, k=2)

        good_points = []
        for m, n in matches:
            if m.distance < 0.6 * n.distance:
                good_points.append(m)

        Lower_keypoints = 0
        if len(kp_1) <= len(kp_2):
            Lower_keypoints = len(kp_1)
        else:
            Lower_keypoints = len(kp_2)

        percentage_similarity = len(good_points) / Lower_keypoints * 100
        print("Similarity: " + str(int(percentage_similarity)) + "%\n")

    else:
        print("----Title: " + titles + "----")
        print('The Images have different dimensions or channels')
        print('Original(X,Y,RGB): ', original.shape)
        print('Compared(X,Y,RGB): ', compared.shape, "\n")
