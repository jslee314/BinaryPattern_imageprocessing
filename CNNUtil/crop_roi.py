import cv2
from CNNUtil import paths

def findRegion(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    rct, thr = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, hierachy = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    len = w if w > h else h
    dst = img[y: y + len, x: x + len]
    return dst








data_dir = 'D:/Data/iris_pattern/test_image/11'

imagePaths = sorted(list(paths.list_images(data_dir)))


for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    image = findRegion(image)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(imagePath)
    cv2.imwrite(imagePath, image)