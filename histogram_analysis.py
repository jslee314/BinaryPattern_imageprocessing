from CNNUtil import img_preprocessing as im_pro
import cv2
from matplotlib import pyplot as plt
import numpy as np

# my_list = np.zeros([10])
# for i in range(5):
#     my = np.array([1, 1, 0, 0, 0, 0, 0, 1, 1, 1])
#     my_list = my_list + my
# my_list = my_list/5
# print(my_list)


def get_normalized_histogram(normal_imgs):
    hist_list = np.zeros([256])
    for i, normal_img in enumerate(normal_imgs):
        img = cv2.cvtColor(normal_img, cv2.COLOR_BGR2GRAY)
        height, width = img.shape[:2]
        # 입력 이미지의 배열/ 히스토그램을 얻을 채널 인덱스/ Mask 이미지/ X 축 요소(BIN)의 개수/ Y 축 요소값의 범위로 하나의 채널에 대한 화소 강도가 0~255이므로 대부분의 경우 [0,256]
        hist = np.array(cv2.calcHist([img], [0], None, [256], [0, 256])) / (height * width)
        hist_list = hist_list + hist
        # plt.plot(hist, 'r')
        # plt.show()
        # ret, thr1 = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # cv2.imshow("original", img)
        # cv2.waitKey(1000)
        #
        # cv2.imshow("BINARY", thr1)
        # cv2.waitKey(1000)
        # cv2.destroyAllWindows()
    hist_list = hist_list / len(normal_imgs)
    return hist_list


normal_img_names, normal_imgs = im_pro.load_images_from_dir('D:/2. data/iris_pattern/Binary/pattern_normal/normal')
pattern_img_names, pattern_imgs = im_pro.load_images_from_dir('D:/2. data/iris_pattern/Binary/pattern_normal/pattern')
defect_img_names, defect_imgs = im_pro.load_images_from_dir('D:/2. data/iris_pattern/Binary/pattern_normal/defect')
lacuna_names, lacuna_imgs = im_pro.load_images_from_dir('D:/2. data/iris_pattern/Binary/pattern_normal/lacuna')
spoke_img_names, spoke_imgs = im_pro.load_images_from_dir('D:/2. data/iris_pattern/Binary/pattern_normal/spoke')
spot_img_names, spot_imgs = im_pro.load_images_from_dir('D:/2. data/iris_pattern/Binary/pattern_normal/spot')

normal_hist_list = get_normalized_histogram(normal_imgs)
pattern_hist_list = get_normalized_histogram(pattern_imgs)

defect_hist_list = get_normalized_histogram(defect_imgs)
lacuna_hist_list = get_normalized_histogram(lacuna_imgs)
spoke_hist_list = get_normalized_histogram(spoke_imgs)
spot_hist_list = get_normalized_histogram(pattern_imgs)



# plt.plot(normal_hist_list[0:150], 'r')
# plt.plot(pattern_hist_list[0:150], 'b')

plt.plot(defect_hist_list[0:150], 'r')
plt.plot(lacuna_hist_list[0:150], 'g')
plt.plot(spoke_hist_list[0:150], 'b')
plt.plot(spot_hist_list[0:150], 'y')
plt.show()


# plt.style.use("ggplot")
# fig, loss_ax = plt.subplots()
# acc_ax = loss_ax.twinx()
# loss_ax.plot(normal_hist_list, 'r')
# acc_ax.plot(pattern_hist_list, 'b')
# loss_ax.set_xlabel('intensity')
# loss_ax.set_ylabel('normal')
# acc_ax.set_ylabel('pattern')
# plt.show()
