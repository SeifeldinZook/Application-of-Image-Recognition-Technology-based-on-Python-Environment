import cv2

original = cv2.imread('/Users/zook/Pycharm Projects/Images/Image Comparison/original.jpg')
compared = cv2.imread('/Users/zook/Pycharm Projects/Images/Image Comparison/original.jpg')  # Note: these images are read as arrays

# ---------------------------------------------------------------------------------------------------------------
''' 1) Check if the images are equal. '''
# ---------------------------------------------------------------------------------------------------------------

or_shape = original.shape            # (y-axis, x-axis, channels) (1079, 1920, 3)
co_shape = compared.shape            # Note: .shape is an array function

if or_shape == co_shape:
    print('The Images have same dimensions and channels')
    difference = cv2.subtract(original, compared)
    # subtract from each pixel of the first image, the value of the corresponding pixel in the second image.

    cv2.imshow('different pixels between Original & Compared', difference)
    b, g, r = cv2.split(difference)
    # cv2.imshow('b', b)
    # cv2.imshow('g', g)
    # cv2.imshow('r', r)

    if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
        # That function counts the no. of elements which isn't 0 in whole image array
        # It can not apply on difference array
        print("And The Channels have same pixels'value!")
    else:
        print("But The Channels have different pixels'value!")
        print('-No.of NonZero Pixels in b channel = ', cv2.countNonZero(b), 'Pixels')
        print('-No.of NonZero Pixels in g channel = ', cv2.countNonZero(g), 'Pixels')
        print('-No.of NonZero Pixels in r channel = ', cv2.countNonZero(r), 'Pixels')
else:
    print('The Images have different dimensions or channels')
    print('Original(X,Y,BGR): ', or_shape)
    print('Compared(X,Y,BGR): ', co_shape)

cv2.imshow("Original", original)   # Take care about the image title, it's essential!
cv2.imshow("Compared", compared)
cv2.waitKey(0)
cv2.destroyAllWindows()
