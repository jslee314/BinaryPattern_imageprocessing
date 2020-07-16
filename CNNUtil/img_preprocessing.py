import numpy as np
import cv2
from CNNUtil import paths
LENGTH = 180

def get_normalized_histogram_list_gray(normal_imgs, intensity=256):
    hist_list = []
    for i, normal_img in enumerate(normal_imgs):
        img = cv2.cvtColor(normal_img, cv2.COLOR_BGR2GRAY)
        height, width = img.shape[:2]
        # 입력 이미지의 배열/ 히스토그램을 얻을 채널 인덱스/ Mask 이미지/ X 축 요소(BIN)의 개수/ Y 축 요소값의 범위로 하나의 채널에 대한 화소 강도가 0~255이므로 대부분의 경우 [0,256]
        hist = np.array(cv2.calcHist([img], [0], None, [256], [0, 256])) / (height * width)
        hist = hist[0:intensity]
        hist_list.append(hist)
    return hist_list


def load_images_from_dir(data_dir):
    imgs = []
    img_names = []
    imagePaths = sorted(list(paths.list_images(data_dir)))

    for imagePath in imagePaths:
        image = cv2.imread(imagePath)
        img_name = imagePath[:-4]
        img_names.append(img_name)
        imgs.append(image)
    return img_names, imgs


def load_images_from_dir(data_dir):
    imgs = []
    img_names = []
    imagePaths = sorted(list(paths.list_images(data_dir)))

    for imagePath in imagePaths:
        image = cv2.imread(imagePath)
        img_name = imagePath[:-4]
        img_names.append(img_name)
        imgs.append(image)
    return img_names, imgs

def load_image_list(data_dir):
    imgs = []
    imagePaths = sorted(list(paths.list_images(data_dir)))
    for imagePath in imagePaths:
        image = cv2.imread(imagePath)
        image = findRegion(image)
        # image = img_padding_2(image)
        image = cv2.resize(image, (180, 180))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        imgs.append(image)
    return imgs

def findRegion_resize(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    rct, thr = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, hierachy = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    # len = w if w > h else h
    # dst = img[y:y+len, x:x+len]
    dst = img[y:y+h, x:x+w]
    return dst

def findRegion(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    rct, thr = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, hierachy = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    x, y, w, h = cv2.boundingRect(contours[0])
    len = w if w > h else h
    dst = img[y : y+len, x: x+len]
    return dst


def img_padding_2(img, LENGTH=180):
    blank_image = np.zeros((LENGTH, LENGTH, 3), np.uint8)
    (w, h) = (img.shape[0], img.shape[1])
    len = w if w > h else h
    if len > LENGTH:
        big_img = np.zeros((len, len, 3), np.uint8)
        big_img[0:  w,  0:  h] = img
        dst = cv2.resize(big_img, (LENGTH, LENGTH))
        blank_image = dst
    else:
        blank_image[0:  w, 0:  h] = img
    return blank_image

# def img_padding(img):
#     # 이미지의 x, y가 300이 넘을 경우 작게해주기
#     blank_image = np.zeros((WIDTH, WIDTH, 3), np.uint8)
#     percent = 1
#     if(img.shape[1] >WIDTH):
#         if (img.shape[1] > img.shape[0]):  # 이미지의 가로가 세보다 크면 가로를 300으로 맞추고 세로를 비율에 맞춰서
#             percent = WIDTH / img.shape[1]
#         else:
#             percent = WIDTH / img.shape[0]
#     if (img.shape[0] > WIDTH):
#         if (img.shape[1] > img.shape[0]):  # 이미지의 가로가 세보다 크면 가로를 300으로 맞추고 세로를 비율에 맞춰서
#             percent = WIDTH / img.shape[1]
#         else:
#             percent = WIDTH / img.shape[0]
#
#     img = cv2.resize(img, dsize=(0, 0), fx=percent, fy=percent, interpolation=cv2.INTER_LINEAR)
#
#     blank_image[ 0: img.shape[0],0: img.shape[1] ] = img
#
#     return blank_image

# data_dir = 'D:\Data\iris_pattern\Original\defect'

data_dir = 'D:/Data/iris_pattern/Binary/defect_binary/train/defect'
# data_dir = 'D:/Data/iris_pattern/test_image/11'
imagePaths = sorted(list(paths.list_images(data_dir)))
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = findRegion(image)
    image = img_padding_2(image, 180)
    # image = findRegion(image)

    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = cv2.resize(image, (HEIGHT, WIDTH))
    # cv2.namedWindow("img_re", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("img_re", 400, 400)
    cv2.imshow("img_re", image)
    # cv2.waitKey(0)