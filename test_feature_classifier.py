from tensorflow.keras.losses import categorical_crossentropy
from CNNUtil import img_preprocessing as im_pro
from CNNUtil.model import Model as model
from CNNUtil import paths
import cv2
import numpy as np

### 1. model load
h5_weight_path = './output/model_weight.h5'
model = model.build_model(intensity=256)
model.summary()
model.load_weights(h5_weight_path)
model.compile(loss=categorical_crossentropy, optimizer='rmsprop', metrics=['accuracy'])

### 2. data load
data_dir = 'D:/2. data/iris_pattern/test_image/11'
imagePaths = sorted(list(paths.list_images(data_dir)))
datas = []

test_img_names, test_imgs = im_pro.load_images_from_dir(data_dir)
normal_hist_list = im_pro.get_normalized_histogram_list_gray(test_imgs)
normal_hist_list = np.squeeze(np.array(normal_hist_list))

### 3. predict
predictions = model.predict(normal_hist_list, batch_size=32)

for i, prediction in enumerate(predictions):
    (normal, pattern) = (prediction[0], prediction[1])
    labels = []
    labels.append("{}: {:.2f}%".format('normal', normal * 100))
    labels.append("{}: {:.2f}%".format('pattern', pattern * 100))

    output = test_imgs[i]
    y = 10

    if normal > 0.5:
        cv2.putText(output, labels[0], (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        print(labels[0])
    else:
        cv2.putText(output, labels[1], (10, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        print(labels[1])

    cv2.imshow("Output", output)
    cv2.waitKey(0)


# import os
# import shutil
#
# src = 'C:/Users/jslee/Downloads/json_20200629_right0147242191_5f900470-0431-4fce-8120-194d1c934fc5.json'
# data_path = 'C:/Users/jslee/Downloads/original/이재선'
# count = 0
# for (root, dirs, files) in os.walk(data_path):
#     for dir in dirs:
#         shutil.copy(src, os.path.join(root, dir) + '/')





