from __future__ import print_function
from __future__ import division
from CNNUtil import paths
import cv2 as cv
import numpy as np

def load_image_list(data_dir):
    imgs = []
    imagePaths = sorted(list(paths.list_images(data_dir)))
    for imagePath in imagePaths:
        image = cv.imread(imagePath)
        # image = img_padding_2(image)
        # image = cv.resize(image, (180, 180))
        # image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        imgs.append(image)
    return imgs


normal_imgs = load_image_list('D:/2. data/iris_pattern/test_image/normal')
pattern_imgs = load_image_list('D:/2. data/iris_pattern/test_image/pattern')

for i, src in enumerate(pattern_imgs):

    ## [Separate the image in 3 places ( B, G and R )]
    bgr_planes = cv.split(src)

    ## [Establish the number of bins]
    histSize = 256

    ## [Set the ranges ( for B,G,R) )]
    histRange = (0, 256)  # the upper boundary is exclusive

    ## [Set histogram param]
    accumulate = False

    ## [Compute the histograms]
    b_hist = cv.calcHist(bgr_planes, [0], None, [histSize], histRange, accumulate=accumulate)
    g_hist = cv.calcHist(bgr_planes, [1], None, [histSize], histRange, accumulate=accumulate)
    r_hist = cv.calcHist(bgr_planes, [2], None, [histSize], histRange, accumulate=accumulate)

    ## [Draw the histograms for B, G and R]
    hist_w = 512
    hist_h = 400
    bin_w = int(round(hist_w / histSize))

    histImage = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)

    ## [Normalize the result to ( 0, histImage.rows )]
    cv.normalize(b_hist, b_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)
    cv.normalize(g_hist, g_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)
    cv.normalize(r_hist, r_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)

    ## [Draw for each channel]
    for i in range(1, histSize):
        cv.line(histImage, (bin_w * (i - 1), hist_h - int(np.round(b_hist[i - 1]))),
                (bin_w * (i), hist_h - int(np.round(b_hist[i]))),
                (255, 0, 0), thickness=2)
        cv.line(histImage, (bin_w * (i - 1), hist_h - int(np.round(g_hist[i - 1]))),
                (bin_w * (i), hist_h - int(np.round(g_hist[i]))),
                (0, 255, 0), thickness=2)
        cv.line(histImage, (bin_w * (i - 1), hist_h - int(np.round(r_hist[i - 1]))),
                (bin_w * (i), hist_h - int(np.round(r_hist[i]))),
                (0, 0, 255), thickness=2)

    ## [Display]
    cv.imshow('Source image', src)
    cv.imshow('calcHist Demo', histImage)
    cv.waitKey()

