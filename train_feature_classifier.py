from tensorflow.keras.losses import sparse_categorical_crossentropy
from CNNUtil import img_preprocessing as im_pro
from CNNUtil.util import Util as utl
from CNNUtil.model import Model as model
from CNNUtil.customcallback import CustomCallback
import numpy as np
from sklearn.metrics import confusion_matrix


h5_path = './output/model.h5'
h5_weight_path = './output/model_weight.h5'

normal_img_names, normal_imgs = im_pro.load_images_from_dir('D:/2. data/iris_pattern/Binary/pattern_normal/normal')
pattern_img_names, pattern_imgs = im_pro.load_images_from_dir('D:/2. data/iris_pattern/Binary/pattern_normal/pattern2')

normal_hist_list = im_pro.get_normalized_histogram_list_gray(normal_imgs)
train_normal_list, test_normal_list = normal_hist_list[:int(len(normal_hist_list) * 0.8)], normal_hist_list[int(len(normal_hist_list) * 0.8):]
pattern_hist_list = im_pro.get_normalized_histogram_list_gray(pattern_imgs)
train_pattern_list, test_pattern_list = pattern_hist_list[:int(len(pattern_hist_list) * 0.8)], pattern_hist_list[int(len(pattern_hist_list) * 0.8):]


x_train_data = train_normal_list + train_pattern_list
x_test_data = test_normal_list + test_pattern_list

x_train_data = np.squeeze(np.array(x_train_data))
x_test_data = np.squeeze(np.array(x_test_data))

y_train_data = np.concatenate((np.zeros(len(train_normal_list)), np.ones(len(train_pattern_list))), axis=0)
y_test_data = np.concatenate((np.zeros(len(test_normal_list)), np.ones(len(test_pattern_list))), axis=0)


model = model.build_model(intensity=256)
model.summary()
model.compile(loss=sparse_categorical_crossentropy, optimizer='rmsprop', metrics=['accuracy'])
list_callbacks = CustomCallback.callback(100, h5_weight_path)

hist = model.fit(x_train_data, y_train_data, epochs=500, validation_data=(x_test_data, y_test_data), callbacks=list_callbacks)


''' save graph '''
utl.hist_saved(hist)

''' save model '''
model.load_weights(h5_weight_path)
model.compile(loss=sparse_categorical_crossentropy, optimizer='rmsprop', metrics=['accuracy'])
model.save(h5_path)

''' save confusion_matrix '''
predictions = model.predict(x_test_data, batch_size=32)
confu_mx = confusion_matrix(y_test_data, predictions.argmax(axis=1))
utl.confusion_matrix_saved(confu_mx)
