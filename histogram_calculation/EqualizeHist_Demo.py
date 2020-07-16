from __future__ import print_function
import cv2 as cv


## [Load image]
data_dir = 'D:/2. data/iris_pattern/test_image/normal/1.png'
src = cv.imread(data_dir)
## [Convert to grayscale]
src = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
## [Apply Histogram Equalization]
dst = cv.equalizeHist(src)
## [Display results]
cv.imshow('Source image', src)
cv.imshow('Equalized Image', dst)

## [Wait until user exits the program]
cv.waitKey()

