# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light,md
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # 1. 개요 
# 개요 및 import와 변수 등에 대한 설정 부분

# + _cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19" _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5"
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import zipfile
import os
import shutil

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.applications import VGG16, DenseNet201
from keras import models
from keras import layers
from keras import optimizers
from enum import Enum
# -

print('케라스 버전 : {}'.format(keras.__version__))
print('텐서플로 버전 : {}'.format(tf.__version__))


# 테스트할 모델 유형을 enum 타입으로 정리.
class ModelType(Enum):
    ALL = 0
    CNN = 1
    VGG16 = 2
    DENSENET201 = 3


# +
# 코드변수들 설정 (코드의 IF 문 같은데서 이용하는 목적임. 훈련,학습과는 관계없음)
IMAGE_UNZIP_WITH_COPY = False

# 검증용, 테스트용 비율
VALID_FILE_RATE = 0.2  # 검증용 파일 비율
TEST_FILE_RATE = 0.2  # 테스트용 파일 비율

# 이미지 가로,세로
IMAGE_WIDTH=150
IMAGE_HEIGHT=150

# 사용할 모델 유형 지정.
MODEL_TYPE = ModelType.DENSENET201
# -

# # 2. 이미지 데이터셋 의 처리
#
# 캐글의 이미지 데이터셋의 압축을 해제하고, 폴더 별로 구성합니다. 
#
# `ImageDataGenerator`를 이용하기 위해서는 train, valid(validation)으로 나눌 필요가 있습니다. ImageDataGenerator의 'validation_split'을 이용할 경우에는 나누지 않아도 됩니다. (단지 이 경우에는 검증값도 이미지가 변형될 수 있음에 주의)

# ## 2.1. 압축 해제

# 1. 최상위 디렉토리 생성. 'train' 디렉토리 생성.
# 2. `extract`에 원본 소스 이미지 압축 풀기
#   - `/input/dogs-vs-cats/` 의 'train.zip'파일을 `/output/kaggle/working/extract` 으로 압축해제 합니다.

# zip 으로 묶여있는 이미지셋 압축 해제
if IMAGE_UNZIP_WITH_COPY : 
    # 폴더가 이미 존재할 경우에는 삭제 후 다시 압축 해제 (그냥 혹시 몰라서 넣는 코드)
    if os.path.exists('./extract'):
        shutil.rmtree('./extract')

    # train.zip압축 해제 (/input/dogs-vs-cats/train.zip)(/output/kaggle/working/train/extract)
    with zipfile.ZipFile("../input/dogs-vs-cats/train.zip", 'r') as zip_ref:
        zip_ref.extractall("./extract")
        
    # test1.zip압축 해제 (/output/kaggle/working)
    #with zipfile.ZipFile("../input/dogs-vs-cats/test1.zip", 'r') as zip_ref:
    #    zip_ref.extractall("./")

# +
# 디렉토리 경로 등을 설정할 변수
dirs = {}
dirs['base'] = './train'
dirs['extract_train'] = './extract/train'
dirs['extract_train_trash'] = os.path.join(dirs['extract_train'], 'trash')
dirs['train'] = os.path.join(dirs['base'], 'train')
dirs['valid'] = os.path.join(dirs['base'], 'valid')
dirs['test'] = os.path.join(dirs['base'], 'test')
train_dir = dirs['train']
valid_dir = dirs['valid']

# 학습용 cats, dogs 
dirs['train_cats'] = os.path.join(dirs['train'], 'cats')
dirs['train_dogs'] = os.path.join(dirs['train'], 'dogs')

# 검증용 cats, dogs
dirs['valid_cats'] = os.path.join(dirs['valid'], 'cats')
dirs['valid_dogs'] = os.path.join(dirs['valid'], 'dogs')

# 테스트용 cats, dogs
dirs['test_cats'] = os.path.join(dirs['test'], 'cats')
dirs['test_dogs'] = os.path.join(dirs['test'], 'dogs')

# +
print("전체 이미지 수 : {}".format(
    len([f for f in os.listdir(dirs['extract_train']) if os.path.isfile(os.path.join(dirs['extract_train'], f))])))

print("고양이 이미지 수 : {}".format(
    len([f for f in os.listdir(dirs['extract_train']) if os.path.isfile(os.path.join(dirs['extract_train'], f)) and 'cat' in f])))

print("강아지 이미지 수 : {}".format(
    len([f for f in os.listdir(dirs['extract_train']) if os.path.isfile(os.path.join(dirs['extract_train'], f)) and 'dog' in f])))
# -

# ## 2.2. 학습, 검증 등을 위한 폴더 생성

if IMAGE_UNZIP_WITH_COPY : 
    # 폴더 및 데이터 재구성
    if os.path.exists(dirs['base']):  # 반복적인 실행을 위해 디렉토리를 삭제합니다.
        shutil.rmtree(dirs['base'])   # 이 코드는 책에 포함되어 있지 않습니다.
    os.mkdir(dirs['base'])

    # train, valid, test 디렉토리 구성
    os.mkdir(dirs['train'])
    os.mkdir(dirs['valid'])
    os.mkdir(dirs['test'])
    
    # 훈련용 cats, dogs 디렉토리
    os.mkdir(dirs['train_cats'])
    os.mkdir(dirs['train_dogs'])
    
    # 검증용 cats, dogs 디렉토리
    os.mkdir(dirs['valid_cats'])
    os.mkdir(dirs['valid_dogs'])

    # 테스트용 cats, dogs 디렉토리
    os.mkdir(dirs['test_dogs'])
    os.mkdir(dirs['test_cats'])

# ## 2.3. 애매한 이미지 파일 제거
# 이 쯤에서 애매하거나 불필요한 이미지파일들은 제거.
# 1. 고양이도 개도 아닌 사진
# 2. 사진이 아닌 그림, 삽화
# 3. 개와 고양이가 둘 다 있는 사진 제외 (학습시 안 좋을 듯 해서 제외)
# 4. 등만 보인다거나 배만 보인다거나 하는 경우 제외
# 5. 신생아 수준의 사진
#
# 참조 링크
# * https://thedatafrog.com/en/articles/dogs-vs-cats/#Pet-cleaning:-improving-the-dataset-quality

# +
bad_cat_ids = [92, 2663, 2939, 3216, 3399, 3731, 4321, 4338, 4503, 4522, 4583, 4688, 4833, 4950,
               5351, 5355, 5418, 5583, 5853, 6215, 6699, 6987, 7362, 7372, 7377, 7564, 7968, 8019, 8420, 8456,
               8470, 9770, 10029, 10151, 10460, 10539, 10570, 10712, 10739, 10905, 11184, 
               11565, 12004, 12272, 12493]

bad_dog_ids = [7, 1043, 1194, 1259, 1308, 1773, 1835, 1895, 2362, 2422,
               2614, 2877, 3322, 4218, 4367, 4706, 5604, 6413, 6475, 6812, 6814,
               7798, 8736, 8898, 9188, 9517, 
               10161, 10190, 10237, 10401, 10747, 10780,
               10797, 10801, 11186, 11191, 11299]

# 이미지 화면에서 확인하기
def plot_images(cate, ids):
    ncols, nrows = 6, 10
    fig = plt.figure( figsize=(ncols*3, nrows*3), dpi=90)
    
    for i, img_id in enumerate(ids):
        if IMAGE_UNZIP_WITH_COPY : 
            dirname = dirs['extract_train']
        else :
            dirname = dirs['extract_train_trash']
        filename = '{}.{}.jpg'.format(cate,img_id)
        img = plt.imread(os.path.join(dirname, filename))
        
        plt.subplot(nrows, ncols, i+1)
        plt.imshow(img)
        plt.title(str(filename))
        plt.axis('off')


bad_cats = plot_images('cat', bad_cat_ids)
bad_dogs = plot_images('dog', bad_dog_ids)


# +
# 제거의 기능을 담당하는 함수
def cleanup(cate, ids):
    oldpwd = os.getcwd()
   
    trash_dir = dirs['extract_train_trash']
    if not os.path.exists(trash_dir):
        os.mkdir(trash_dir)
    
    for id in ids:
        filename = '{cate}.{id}.jpg'.format(cate=cate,id=id)
        src = os.path.join(dirs['extract_train'], filename)
        shutil.move(src, trash_dir)

# 제거 실행
if IMAGE_UNZIP_WITH_COPY : 
    cleanup('cat', bad_cat_ids)
    cleanup('dog', bad_dog_ids)
# -

# ## 2.4. 각 폴더별로 이미지 나눠서 복사
# 각각 train, validation, test 폴더로 나눠서 구성한다. 
# 전체 이미지의 갯수는 25,000이고, 고양이 강아지 각각 반으로 나누면 12,500 개씩 있을 것이고. 60%, 20%, 20% 으로 구성한다면. (참고 https://ebbnflow.tistory.com/125)
#
# 예상치 (빠진 것들이 있으므로 이보다 적음)
# * train - 7,500
# * validation - 2,500
# * test - 2,500 

# +
# 비율별로 추출할 번호를 계산
cat_count_total = 12500
dog_count_total = 12500

cat_count_valid = int(cat_count_total * VALID_FILE_RATE)
cat_count_test = int(cat_count_total * TEST_FILE_RATE)
cat_count_train = cat_count_total - cat_count_valid - cat_count_test

dog_count_valid = int(dog_count_total * VALID_FILE_RATE)
dog_count_test = int(dog_count_total * TEST_FILE_RATE)
dog_count_train = dog_count_total - dog_count_valid - dog_count_test

print("cats : total {} / train {} / valid {} / test {}".format(cat_count_total, cat_count_train, cat_count_valid, cat_count_test))
print("dogs : total {} / train {} / valid {} / test {}".format(dog_count_total, dog_count_train, dog_count_valid, dog_count_test))


# +
# 파일명으로 구분하여 복사를 실행하는 함수
def partialCopyImages(cate, start, end, origin, dest) :
    files = ['{}.{}.jpg'.format(cate, i) for i in range(start, end)]
    for filename in files:
        src = os.path.join(origin, filename)
        dst = os.path.join(dest, filename)
        if IMAGE_UNZIP_WITH_COPY : 
            if os.path.isfile(src):
                shutil.copyfile(src, dst)

if IMAGE_UNZIP_WITH_COPY : 
    origin_dir = dirs['extract_train']
    
    # 고양이 [학습용, 검증용, 테스트용] 복사
    partialCopyImages('cat', 0, cat_count_train, origin_dir, dirs['train_cats'])
    partialCopyImages('cat', cat_count_train, cat_count_train+cat_count_valid, origin_dir, dirs['valid_cats'])
    partialCopyImages('cat', cat_count_train+cat_count_valid, cat_count_total, origin_dir, dirs['test_cats'])

    # 강아지 [학습용, 검증용, 테스트용] 복사
    partialCopyImages('dog', 0, dog_count_train, origin_dir, dirs['train_dogs'])
    partialCopyImages('dog', dog_count_train, dog_count_train+dog_count_valid, origin_dir, dirs['valid_dogs'])
    partialCopyImages('dog', dog_count_train+dog_count_valid, dog_count_total, origin_dir, dirs['test_dogs'])
# -

# 결과 확인
print('고양이 이미지 [학습용]: ', len(os.listdir(dirs['train_cats'])))
print('고양이 이미지 [검증용]: ', len(os.listdir(dirs['valid_cats'])))
print('고양이 이미지 [테스트용] ', len(os.listdir(dirs['test_cats'])))
print('--------------------------------')
print('강아지 이미지 [학습용] ', len(os.listdir(dirs['train_dogs'])))
print('강아지 이미지 [검증용] ', len(os.listdir(dirs['valid_dogs'])))
print('강앙지 이미지 [테스트용] ', len(os.listdir(dirs['test_dogs'])))

# # 3. 모델 종류별

# # 3.1. CNN 모델
# 간단히 작성한 CNN 모델로 어떤 성능을 나타내는지 확인해보자.

# ## 3.1.1. 데이터 전처리 (CNN)

if MODEL_TYPE == ModelType.CNN or MODEL_TYPE == ModelType.ALL:
    # ------------------------
    # 훈련용 데이터 전처리
    # ------------------------
    train_dategen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
    )

    # batch_size = 128
    batch_size = 20

    train_generator = train_dategen.flow_from_directory(
        train_dir,
        target_size = (IMAGE_WIDTH,IMAGE_HEIGHT),
        batch_size = batch_size,
        class_mode='binary')


    # ------------------------
    # 검증용 데이터 전처리
    # ------------------------
    valid_datagen = ImageDataGenerator(rescale=1./255)
    valid_generator = valid_datagen.flow_from_directory(
        valid_dir,
        target_size = (IMAGE_WIDTH,IMAGE_HEIGHT),
        batch_size = batch_size,
        class_mode='binary')

    # ------------------------
    # 데이터 형태 확인
    # ------------------------
    for data_batch, labels_batch in train_generator:
        print('Data batch shape:', data_batch.shape)
        print('Labels batch shape:', labels_batch.shape)
        break;

# ## 3.1.2. 모델 생성 (CNN)

if MODEL_TYPE == ModelType.CNN or MODEL_TYPE == ModelType.ALL:
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                            input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.summary()

# ```
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# conv2d (Conv2D)              (None, 148, 148, 32)      896       
# _________________________________________________________________
# max_pooling2d (MaxPooling2D) (None, 74, 74, 32)        0         
# _________________________________________________________________
# conv2d_1 (Conv2D)            (None, 72, 72, 64)        18496     
# _________________________________________________________________
# max_pooling2d_1 (MaxPooling2 (None, 36, 36, 64)        0         
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 34, 34, 128)       73856     
# _________________________________________________________________
# max_pooling2d_2 (MaxPooling2 (None, 17, 17, 128)       0         
# _________________________________________________________________
# conv2d_3 (Conv2D)            (None, 15, 15, 128)       147584    
# _________________________________________________________________
# max_pooling2d_3 (MaxPooling2 (None, 7, 7, 128)         0         
# _________________________________________________________________
# flatten (Flatten)            (None, 6272)              0         
# _________________________________________________________________
# dense (Dense)                (None, 512)               3211776   
# _________________________________________________________________
# dense_1 (Dense)              (None, 1)                 513       
# =================================================================
# Total params: 3,453,121
# Trainable params: 3,453,121
# Non-trainable params: 0
# ```

# ## 3.1.3. 모델 학습 (CNN)

if MODEL_TYPE == ModelType.CNN or MODEL_TYPE == ModelType.ALL:
    model.compile(loss='binary_crossentropy',
             optimizer=keras.optimizers.Adamax(lr=0.001),
             metrics=['acc'])
    
    # validation_steps=50,
    history = model.fit(
        train_generator,
        validation_data=valid_generator,
        epochs = 50,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True)]
    )

# ```
# Epoch 1/50
# 748/748 [==============================] - 122s 162ms/step - loss: 0.4302 - acc: 0.8035 - val_loss: 0.3871 - val_acc: 0.8356
# Epoch 2/50
# 748/748 [==============================] - 122s 163ms/step - loss: 0.4033 - acc: 0.8199 - val_loss: 0.3175 - val_acc: 0.8649
# Epoch 3/50
# 748/748 [==============================] - 125s 167ms/step - loss: 0.3677 - acc: 0.8354 - val_loss: 0.3125 - val_acc: 0.8609
# Epoch 4/50
# 748/748 [==============================] - 126s 169ms/step - loss: 0.3475 - acc: 0.8489 - val_loss: 0.2795 - val_acc: 0.8801
# Epoch 5/50
# 748/748 [==============================] - 125s 167ms/step - loss: 0.3208 - acc: 0.8599 - val_loss: 0.2689 - val_acc: 0.8811
# Epoch 6/50
# 748/748 [==============================] - 125s 168ms/step - loss: 0.3156 - acc: 0.8671 - val_loss: 0.3007 - val_acc: 0.8681
# Epoch 7/50
# 748/748 [==============================] - 125s 168ms/step - loss: 0.2953 - acc: 0.8728 - val_loss: 0.3719 - val_acc: 0.8448
# Epoch 8/50
# 748/748 [==============================] - 124s 166ms/step - loss: 0.2937 - acc: 0.8746 - val_loss: 0.2497 - val_acc: 0.8887
# Epoch 9/50
# 748/748 [==============================] - 125s 168ms/step - loss: 0.2827 - acc: 0.8814 - val_loss: 0.2377 - val_acc: 0.9018
# Epoch 10/50
# 748/748 [==============================] - 124s 166ms/step - loss: 0.2791 - acc: 0.8765 - val_loss: 0.2589 - val_acc: 0.8879
# Epoch 11/50
# 748/748 [==============================] - 125s 168ms/step - loss: 0.2605 - acc: 0.8894 - val_loss: 0.2760 - val_acc: 0.8871
# Epoch 12/50
# 748/748 [==============================] - 124s 166ms/step - loss: 0.2547 - acc: 0.8893 - val_loss: 0.2268 - val_acc: 0.8986
# Epoch 13/50
# 748/748 [==============================] - 123s 164ms/step - loss: 0.2581 - acc: 0.8878 - val_loss: 0.2008 - val_acc: 0.9146
# Epoch 14/50
# 748/748 [==============================] - 122s 163ms/step - loss: 0.2307 - acc: 0.9026 - val_loss: 0.2420 - val_acc: 0.8988
# Epoch 15/50
# 748/748 [==============================] - 121s 162ms/step - loss: 0.2309 - acc: 0.9018 - val_loss: 0.2103 - val_acc: 0.9164
# Epoch 16/50
# 748/748 [==============================] - 121s 161ms/step - loss: 0.2251 - acc: 0.9058 - val_loss: 0.1971 - val_acc: 0.9170
# Epoch 17/50
# 748/748 [==============================] - 120s 160ms/step - loss: 0.2257 - acc: 0.9035 - val_loss: 0.2086 - val_acc: 0.9120
# Epoch 18/50
# 748/748 [==============================] - 119s 159ms/step - loss: 0.2242 - acc: 0.9040 - val_loss: 0.2051 - val_acc: 0.9140
# Epoch 19/50
# 748/748 [==============================] - 120s 160ms/step - loss: 0.2099 - acc: 0.9099 - val_loss: 0.1685 - val_acc: 0.9290
# Epoch 20/50
# 748/748 [==============================] - 119s 159ms/step - loss: 0.2158 - acc: 0.9114 - val_loss: 0.2015 - val_acc: 0.9168
# Epoch 21/50
# 748/748 [==============================] - 120s 160ms/step - loss: 0.2094 - acc: 0.9150 - val_loss: 0.1732 - val_acc: 0.9292
# Epoch 22/50
# 748/748 [==============================] - 119s 160ms/step - loss: 0.1995 - acc: 0.9132 - val_loss: 0.1956 - val_acc: 0.9174
# Epoch 23/50
# 748/748 [==============================] - 119s 159ms/step - loss: 0.2005 - acc: 0.9165 - val_loss: 0.1935 - val_acc: 0.9178
# Epoch 24/50
# 748/748 [==============================] - 119s 159ms/step - loss: 0.1863 - acc: 0.9190 - val_loss: 0.1617 - val_acc: 0.9360
# Epoch 25/50
# 748/748 [==============================] - 121s 161ms/step - loss: 0.1830 - acc: 0.9236 - val_loss: 0.2323 - val_acc: 0.9022
# Epoch 26/50
# 748/748 [==============================] - 121s 162ms/step - loss: 0.1759 - acc: 0.9266 - val_loss: 0.1771 - val_acc: 0.9276
# Epoch 27/50
# 748/748 [==============================] - 121s 162ms/step - loss: 0.1853 - acc: 0.9217 - val_loss: 0.1882 - val_acc: 0.9194
# Epoch 28/50
# 748/748 [==============================] - 124s 166ms/step - loss: 0.1817 - acc: 0.9253 - val_loss: 0.1614 - val_acc: 0.9328
# Epoch 29/50
# 748/748 [==============================] - 122s 162ms/step - loss: 0.1763 - acc: 0.9275 - val_loss: 0.1654 - val_acc: 0.9324
# Epoch 30/50
# 748/748 [==============================] - 121s 161ms/step - loss: 0.1719 - acc: 0.9290 - val_loss: 0.1851 - val_acc: 0.9246
# Epoch 31/50
# 748/748 [==============================] - 120s 161ms/step - loss: 0.1624 - acc: 0.9348 - val_loss: 0.1693 - val_acc: 0.9340
# Epoch 32/50
# 748/748 [==============================] - 121s 162ms/step - loss: 0.1667 - acc: 0.9291 - val_loss: 0.1540 - val_acc: 0.9356
# Epoch 33/50
# 748/748 [==============================] - 122s 163ms/step - loss: 0.1696 - acc: 0.9313 - val_loss: 0.1947 - val_acc: 0.9166
# Epoch 34/50
# 748/748 [==============================] - 121s 162ms/step - loss: 0.1682 - acc: 0.9293 - val_loss: 0.1864 - val_acc: 0.9238
# Epoch 35/50
# 748/748 [==============================] - 121s 162ms/step - loss: 0.1560 - acc: 0.9360 - val_loss: 0.1445 - val_acc: 0.9419
# Epoch 36/50
# 748/748 [==============================] - 120s 160ms/step - loss: 0.1544 - acc: 0.9338 - val_loss: 0.1589 - val_acc: 0.9344
# Epoch 37/50
# 748/748 [==============================] - 120s 161ms/step - loss: 0.1534 - acc: 0.9366 - val_loss: 0.1798 - val_acc: 0.9296
# Epoch 38/50
# 748/748 [==============================] - 120s 161ms/step - loss: 0.1536 - acc: 0.9368 - val_loss: 0.1869 - val_acc: 0.9282
# Epoch 39/50
# 748/748 [==============================] - 120s 160ms/step - loss: 0.1565 - acc: 0.9355 - val_loss: 0.1475 - val_acc: 0.9387
# Epoch 40/50
# 748/748 [==============================] - 120s 161ms/step - loss: 0.1591 - acc: 0.9355 - val_loss: 0.1677 - val_acc: 0.9381
# Epoch 41/50
# 748/748 [==============================] - 120s 161ms/step - loss: 0.1548 - acc: 0.9386 - val_loss: 0.1349 - val_acc: 0.9441
# Epoch 42/50
# 748/748 [==============================] - 121s 161ms/step - loss: 0.1490 - acc: 0.9413 - val_loss: 0.1504 - val_acc: 0.9368
# Epoch 43/50
# 748/748 [==============================] - 121s 161ms/step - loss: 0.1564 - acc: 0.9353 - val_loss: 0.1577 - val_acc: 0.9360
# Epoch 44/50
# 748/748 [==============================] - 121s 162ms/step - loss: 0.1492 - acc: 0.9385 - val_loss: 0.1455 - val_acc: 0.9397
# Epoch 45/50
# 748/748 [==============================] - 120s 160ms/step - loss: 0.1302 - acc: 0.9469 - val_loss: 0.1809 - val_acc: 0.9264
# Epoch 46/50
# 748/748 [==============================] - 121s 161ms/step - loss: 0.1310 - acc: 0.9476 - val_loss: 0.1589 - val_acc: 0.9372
# Epoch 47/50
# 748/748 [==============================] - 120s 161ms/step - loss: 0.1414 - acc: 0.9452 - val_loss: 0.1493 - val_acc: 0.9370
# Epoch 48/50
# 748/748 [==============================] - 120s 160ms/step - loss: 0.1396 - acc: 0.9456 - val_loss: 0.1548 - val_acc: 0.9368
# Epoch 49/50
# 748/748 [==============================] - 120s 161ms/step - loss: 0.1297 - acc: 0.9465 - val_loss: 0.1580 - val_acc: 0.9389
# Epoch 50/50
# 748/748 [==============================] - 120s 161ms/step - loss: 0.1375 - acc: 0.9448 - val_loss: 0.1340 - val_acc: 0.9501
# ```

# ## 3.1.4. 모델 저장 (CNN)

# 모델 파일 저장하기.
if MODEL_TYPE == ModelType.CNN or MODEL_TYPE == ModelType.ALL:
    model.save('cats_vs_dogs_cnn.h5')


def plot_history(history, yrange):
    '''Plot loss and accuracy as a function of the epoch,
    for the training and validation datasets.
    '''
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Get number of epochs
    epochs = range(len(acc))

    # Plot training and validation accuracy per epoch
    plt.plot(epochs, acc)
    plt.plot(epochs, val_acc)
    plt.title('Training and validation accuracy')
    plt.ylim(yrange)
    
    # Plot training and validation loss per epoch
    plt.figure()

    plt.plot(epochs, loss)
    plt.plot(epochs, val_loss)
    plt.title('Training and validation loss')
    
    plt.show()


if MODEL_TYPE == ModelType.CNN or MODEL_TYPE == ModelType.ALL:
    plot_history(history, (0.65, 1.))

# ![cnn-graph](data:image/png;base64,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)
# ![cnn-graph-2](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAXoAAAEICAYAAABRSj9aAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjQuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/Z1A+gAAAACXBIWXMAAAsTAAALEwEAmpwYAABHkklEQVR4nO3dd1iUV/bA8e+hgyKIIiqIimKPFWvUoKaYpqZtTDW9t03blE3f5JdNNmV3Y7opu4kxrmkmMTHGlmhs2HvDgtgLCBYQuL8/7oyOMMBQB4bzeR6fmXnnLfcFPPPOee89V4wxKKWU8l1+3m6AUkqpqqWBXimlfJwGeqWU8nEa6JVSysdpoFdKKR+ngV4ppXycBnpVJiLyk4iMqex1vUlEtorI2VWwXyMibR3P3xWRpzxZtxzHuUZEfilvO0vYb7KI7Kjs/arqF+DtBqiqJyLZLi/DgBwg3/H6dmPM557uyxhzflWs6+uMMXdUxn5EpBWwBQg0xuQ59v054PHvUNU9GujrAGNMfedzEdkK3GKM+bXweiIS4AweSinfoambOsz51VxE/iIiu4GPRaShiPwgIvtE5JDjeZzLNrNE5BbH8xtEZI6I/MOx7hYROb+c67YWkd9EJEtEfhWRsSLyWTHt9qSNL4jIXMf+fhGRxi7vXyci20TkgIg8WcLPp6+I7BYRf5dll4jICsfzPiIyT0QyRGSXiLwlIkHF7OsTEfmby+tHHNvsFJGbCq17oYgsFZHDIpImIs+6vP2b4zFDRLJFpL/zZ+uy/QARWSQimY7HAZ7+bEoiIh0d22eIyGoRGeHy3gUissaxz3QRedixvLHj95MhIgdF5HcR0bhTzfQHrpoCUUBL4Dbs38THjtfxwDHgrRK27wusBxoDrwDjRETKse54YCHQCHgWuK6EY3rSxquBG4EmQBDgDDydgHcc+2/uOF4cbhhjFgBHgKGF9jve8Twf+LPjfPoDw4C7Smg3jjYMd7TnHCARKHx/4AhwPRAJXAjcKSKjHO8NdjxGGmPqG2PmFdp3FPAj8C/Hub0O/CgijQqdQ5GfTSltDgS+B35xbHcv8LmItHesMg6bBgwHugAzHMsfAnYA0UAM8ASgdVeqmQZ6VQA8Y4zJMcYcM8YcMMZ8ZYw5aozJAl4Eziph+23GmA+MMfnAp0Az7H9oj9cVkXigN/C0MSbXGDMHmFzcAT1s48fGmA3GmGPARKC7Y/nlwA/GmN+MMTnAU46fQXG+AK4CEJFw4ALHMowxi40x840xecaYrcB7btrhzp8c7VtljDmC/WBzPb9ZxpiVxpgCY8wKx/E82S/YD4aNxpj/Otr1BbAOuNhlneJ+NiXpB9QHXnb8jmYAP+D42QAngE4i0sAYc8gYs8RleTOgpTHmhDHmd6MFtqqdBnq1zxhz3PlCRMJE5D1HauMwNlUQ6Zq+KGS384kx5qjjaf0yrtscOOiyDCCtuAZ72MbdLs+PurSpueu+HYH2QHHHwl69XyoiwcClwBJjzDZHO9o50hK7He14CXt1X5rT2gBsK3R+fUVkpiM1lQnc4eF+nfveVmjZNiDW5XVxP5tS22yMcf1QdN3vZdgPwW0iMltE+juWvwpsAn4RkVQRecyz01CVSQO9Knx19RDQHuhrjGnAqVRBcemYyrALiBKRMJdlLUpYvyJt3OW6b8cxGxW3sjFmDTagnc/paRuwKaB1QKKjHU+Upw3Y9JOr8dhvNC2MMRHAuy77Le1qeCc2peUqHkj3oF2l7bdFofz6yf0aYxYZY0Zi0zrfYr8pYIzJMsY8ZIxJAEYAD4rIsAq2RZWRBnpVWDg2553hyPc+U9UHdFwhpwDPikiQ42rw4hI2qUgbJwEXichAx43T5yn9/8F44H7sB8r/CrXjMJAtIh2AOz1sw0TgBhHp5PigKdz+cOw3nOMi0gf7AeO0D5tqSihm31OAdiJytYgEiMiVQCdsmqUiFmCv/h8VkUARScb+jiY4fmfXiEiEMeYE9mdSACAiF4lIW8e9mEzsfY2SUmWqCmigV4W9CYQC+4H5wM/VdNxrsDc0DwB/A77E9vd3503K2UZjzGrgbmzw3gUcwt4sLIkzRz7DGLPfZfnD2CCcBXzgaLMnbfjJcQ4zsGmNGYVWuQt4XkSygKdxXB07tj2KvScx19GTpV+hfR8ALsJ+6zkAPApcVKjdZWaMycUG9vOxP/e3geuNMescq1wHbHWksO7A/j7B3mz+FcgG5gFvG2NmVqQtquxE74uomkhEvgTWGWOq/BuFUr5Or+hVjSAivUWkjYj4ObofjsTmepVSFaQjY1VN0RT4GntjdAdwpzFmqXebpJRv0NSNUkr5OE3dKKWUj6txqZvGjRubVq1aebsZSilVqyxevHi/MSba3Xs1LtC3atWKlJQUbzdDKaVqFREpPCL6JE3dKKWUj9NAr5RSPk4DvVJK+TgN9Eop5eM00CullI/TQK+UUj5OA71SSvk4nwn0GUdzefPXDazdddjbTVFKqRrFZwK9ILw9czOTFpdWWlwppeoWnwn0EWGBJLeP5vvlO8kv0EJtSinl5DOBHmBk91j2ZuUwP7WkuZ6VUqpu8alAP6xjE+oHB/Dt0orOg6yUUr7DpwJ9SKA/53Vuys+rdnP8RL63m6OUUjWCR4FeRIaLyHoR2SQij5Ww3mUiYkQkyWXZ447t1ovIeZXR6JKM6tGcrJw8Zq7bW9WHUkqpWqHUQC8i/sBY7OzvnYCrRKSTm/XCgfuBBS7LOgGjgc7AcOBtx/6qTP+ERjSuH8x3y3ZW5WGUUqrW8OSKvg+wyRiTaozJBSZgJ24u7AXg78Bxl2UjgQnGmBxjzBZgk2N/VSbA34+LuzVjxrq9ZB47UZWHUkqpWsGTQB8LpLm83uFYdpKI9ARaGGN+LOu2ju1vE5EUEUnZt2+fRw0vycjuseTmFzB11e4K70sppWq7Ct+MFRE/4HXgofLuwxjzvjEmyRiTFB3tdiasMukWF0GrRmF8u0x73yillCeBPh1o4fI6zrHMKRzoAswSka1AP2Cy44ZsadtWCRFhZPdY5qUeYHfm8dI3UEopH+ZJoF8EJIpIaxEJwt5cnex80xiTaYxpbIxpZYxpBcwHRhhjUhzrjRaRYBFpDSQCCyv9LNwY2b05xsAPK/SmrFKqbis10Btj8oB7gKnAWmCiMWa1iDwvIiNK2XY1MBFYA/wM3G2MqZYO7gnR9ekaF6HpG6VUnRfgyUrGmCnAlELLni5m3eRCr18EXixn+ypkRLfm/O3HtWzam03bJvW90QSllPI6nxoZW9iIbs0Rgcl6Va+UqsN8OtA3aRDCgDaN+HbZTozRipZKqbrJdwJ97lFY/xMc3HLa4pHdY9l+8CjL0jK80y6llPIy3wn0J47CF6Nhw8+nLR7epSlBAX5MXq69b5RSdZPvBPqwRhAYBhnbT1vcICSQAW0aMXt9xUfcKqVUbeQ7gV4EIlsWCfQAgxKjSd1/hLSDR73QMKWU8i7fCfQAkfGQsa3I4sGJjQGYs2l/dbdIKaW8zgcDfdEr+rZN6tO0QQi/b9T0jVKq7vG9QH88E45lnLZYRBiU2Ji5mw7oxOFKqTrH9wI9QGZakbcGtYsm89gJVqZnVnOjlFLKu3wz0LtJ3wxs2xgR+H2Dpm+UUnWLjwX6lvbRTaCPqhdEl+YR/L5Rb8gqpeoW3wr0YVEQWM9toAcYlNiYJdsPkXVcpxhUStUdvhXoRYrteQMwMLExeQWG+akHq69N2+bB7pXVdzyllCrEtwI92EB/qGhfeoBeLRsSGuhfvd0sJ98DM7xSpVkppQBfDPQN3Y+OBQgO8KdfQhRzqitPX1Bg23JU7wsopbzH9wJ9ZDzkFO1L71St5RCO7IX8XDh2qOqPpZRSxfDNQA/FXtUPbleN5RCcbdBAr5TyojoX6NtE16dZRDWVQ3AN9AUFVX88pZRywwcDffF96eFUOYQ5G/dXfTkE5whdUwA5h6v2WEopVQzfC/ShDSGofrGBHmye/vDxPFbsyKjatri2QdM3Sikv8b1AX0pfeoAzneUQqrr3TYZLzR0N9EopL/G9QA+lBnpnOYQq72aZmQb1ou3zY9U4SEsppVzUyUAP1VAOwRjbhmbd7OtiunsqpVRV891An5NZYrpkUGJ01ZZDOHrQTlh+MtBr6kYp5R0+GuhL7nkD0LNlJGFBVVgOIdNx7KZn2EcN9EopL/HRQF9yX3qw5RAGtGnElJW7OJCdU/ltcB47KgGCG9grfKWU8gKPAr2IDBeR9SKySUQec/P+HSKyUkSWicgcEenkWN5KRI45li8TkXcr+wTc8iDQAzx0bnsOH8/jof8tp6Cy+9Q7e9xEtIDQSL2iV0p5TamBXkT8gbHA+UAn4CpnIHcx3hhzhjGmO/AK8LrLe5uNMd0d/+6opHaXLLQhBIWXGug7NmvAUxd1Ytb6fXw0d0vltiEzzfbnD20IoVEa6JVSXuPJFX0fYJMxJtUYkwtMAEa6rmCMcR32WQ/w7gzcHvSld7q2bzzndY7h7z+vY3laRuW1IWO7bYOIDfYa6JVSXuJJoI8FXGfb3uFYdhoRuVtENmOv6O9zeau1iCwVkdkiMsjdAUTkNhFJEZGUffsq6eaoh4FeRHjlsm40CQ/h3i+WVl53y4w0m7YBR6DXHL1Syjsq7WasMWasMaYN8Bfgr47Fu4B4Y0wP4EFgvIg0cLPt+8aYJGNMUnR0dOU0yBnoTelfLiLCAvnn6O6kZxzjyW9WYTzYplSZ2yHSNdDrFb1Syjs8CfTpQAuX13GOZcWZAIwCMMbkGGMOOJ4vBjYD7crV0rKKjLeFxI5neLR6Uqso/nx2IpOX7+R/i3dU7NjHM+0/503hsCitYKmU8hpPAv0iIFFEWotIEDAamOy6gogkury8ENjoWB7tuJmLiCQAiUBqZTS8VB72vHF1Z3JbBrRpxDPfrWbT3qzyH9u1xw3YK3qtYKmU8pJSA70xJg+4B5gKrAUmGmNWi8jzIjLCsdo9IrJaRJZhUzRjHMsHAyscyycBdxhjqidZXY5A7+8nvHFld8KC/Lln/FLy8st5Be4sT+xsQ2hD+6jpG6WUFwR4spIxZgowpdCyp12e31/Mdl8BX1WkgeXW0DE6tpiJwosT0yCE50d24e7xS5i6eg8Xdm1W9mM7P1xOBvoo+3jsENC67PtTSqkK8M2RsQAhkXZEahmu6J2Gd2lKfFQY4+aUM8uUsR0CQk5VrtQreqWUF/luoC9DX/rC/P2EG89sxZLtGSzdXo7gnJkGEXG2DaCBXinlVb4b6KHcgR7giqQWhAcHMG5OOUbMOgdLOWmgV0p5Ud0I9OXoF18/OIDRfVrw06rdpGccK9vGroOlQAO9UsqrfD/Q52aVO8COGdAKYwz/+WOr5xvlHoWj+0+/ovcP0AqWSimv8f1AD+VO38Q1DOP8Ls0Yv3A7R3LyPNuocNdKJ61gqZTyEg30pbhpYGuyjucxydPRsoUHSzlpGQSllJdooC9Fr5YN6d4iko/nbvGsZn1moT70TlqqWCnlJb4d6CvQl97VzQNbs/XAUaav21v6yhnbwS8AwpuevlwrWCqlvMS3A72InT82o2yjYws7v0tTmkeEeDaAKiMNGsSCn//pyzV1o5TyEt8O9FChvvROAf5+jBnQivmpB1m9M7PklTPTiqZt4FSg1wqWSqlqVncCfQVrzI/uE09YkH/pA6gKD5ZyCouyFSxzK1AVUymlyqFuBPrc7AqnTSJCA7miVxzfL9/J1v1H3K+UlwtZu4v2uIFTg6a0L71SqprVjUAPFc7TA9wyKIGwoAAuf3ceK3ZkFF3h8A7AFJ+6Ac3TK6WqXR0K9BXL0wO0iArjqzv7ExLox5XvzWf62j2nr3CyPLG7K3rXUsVKKVV9NNCXUdsm4Xx91wDaNqnPrf9J4fMFLt8UihssBXpFr5TyGt8P9KGREBwBm2fC8cqZyq9JeAgTbutHcvsmPPnNKv7+8zo7mCozDcTPdq8s0g4N9Eop7/D9QA/Q93bYPB3eSoKln1dKF8d6wQG8f10vru4bzzuzNvPnicvIP7gNwptBQFDRDUIj7aMGeqVUNasbgX7ok3DrDJvG+e4uGHc27Eip8G4D/P14cVQXHh3enu+W7WTr5nXu0zYA/oF2lK4GeqVUNasbgR4gthfc9Atc8h5kpsOHw+Dr2213yAoQEe5Kbss9Q9oSlJ3ODqKLXzk0UrtXKqWqXd0J9AB+ftBtNNybAgMfhNVfw+eXV8qu7x/amuZ+B5iSFkDawaPuV9IyCEopL6hbgd4pOBzOfgaGPQ27V0KmhyWISxB4ZA/+FLDTRPPAl8vIy3dzH0ADvVLKC+pmoHdKGGIfU2dXfF+OCUfOPbM3i7cd4l/TNxZdR0sVK6W8oG4H+iadoF40bKmEQO/opz+gZw8u6xnHWzM3sSD1wOnraKlipZQX1O1A7+cHrc+C1FkVLnp2arBUHM+N7Ex8VBgPfLmMjKO5p9bRCpZKKS+o24EeICEZsvfAvnUV20/mdqjXBAJDqR8cwL+u6sH+7Bwe+2olxvkhEtpQK1gqpaqdR4FeRIaLyHoR2SQij7l5/w4RWSkiy0Rkjoh0cnnvccd260XkvMpsfKVIOMs+ps6q2H4ytp9W46ZrXCQPn9uen1fvZsIix9V+mKPejXaxVEpVo1IDvYj4A2OB84FOwFWugdxhvDHmDGNMd+AV4HXHtp2A0UBnYDjwtmN/NUdkPEQlVPyGbEbRCUduHZTAmW0b8eKPa9mZcUzLICilvMKTK/o+wCZjTKoxJheYAIx0XcEY41pEph7gTHiPBCYYY3KMMVuATY791SwJybB1DuSfKN/2BQW2i2ahUbF+fsLLl3Ylv8Dw1LerMCGR9g0N9EqpauRJoI8F0lxe73AsO42I3C0im7FX9PeVcdvbRCRFRFL27dvnadsrT0KyzZunLynf9rtXQH4OxHQu8laLqDAeOrcd09ftZdaOfLtQA71SqhpV2s1YY8xYY0wb4C/AX8u47fvGmCRjTFJ0dAklBKpKq0GAlD9Pv3mGfXT2yy/khgGt6BoXwUszHOUWNNArpaqRJ4E+HXDNScQ5lhVnAjCqnNt6R1gUNOtWsUAfcwaEx7h9O8Dfj5cv7cr2Y46qlhrolVLVyJNAvwhIFJHWIhKEvbk62XUFEUl0eXkh4BwWOhkYLSLBItIaSAQWVrzZVSAhGXYsgpzssm2Xkw3b50Mb91fzTp2aN+Dmwe3IMqGk79pZ/nYqpVQZlRrojTF5wD3AVGAtMNEYs1pEnheREY7V7hGR1SKyDHgQGOPYdjUwEVgD/AzcbYzJr/zTqAQJyVBwArbPK9t22+ba7doMLXXV+4Ylku0XzsqNWziWWzN/DEop3xPgyUrGmCnAlELLnnZ5fn8J274IvFjeBlab+H7gH2zTN4nneL7d5hkQEArx/UtdNSTQn/CGTQjcl8kbv27giQs6lr+9SinlIR0Z6xQYaoN9WfP0m2dAqzMhMMSj1etHNqZt+Ak+/D2VlTsyy95OpZQqIw30rhLOgj2rINvDLp4ZabB/g0dpm5NCGxIXcpxG9YP588Rl7M/OKV9blVLKQxroXSUk20dPq1mmzrSPZQr0UfgfP8Q/r+zOjkNH+dN789iVeaxMzVRKqbLQQO+qWXcIifA8fbNpOoQ3h+gOnh/DUcFyQEIU/7mpL/sO53D5O/PYduBIeVqslFKl0kDvys8fWg/2rGxxQb5dr81QEPH8GC4VLPu0jmL8rf04mpvHFe/OY8MerWqplKp8GugLS0i2s0UdTC15vZ3L4HhGqf3niyhU2OyMuAgm3m577PzpvXms2JFRtv0ppVQpNNAX5ixjUFqefvMMQIote1AsN6WKE2PCmXTHAOoHB3D1BwuKzkyllFIVoIG+sKgEaBBXep5+8wxbNqFeo7Ltv5hSxfGNwph0xwBiGgQz5uOFmsZRSlUaDfSFidj0TepsyD3qfp3jh2HHwrL1tnEqoSZ904gQvritH2FBATw4cRkn8nXKQaVUxWmgd6fbaDieCZPvdX9TduvvUJBXzkDvSN0UU9isSXgIL13ShVXph3lrxqay718ppQrRQO9O60Ew7ClYNQnmvln0/c0zILAetOhb9n2HRtrHEipYDu/SjFHdm/PWzE2e3Zw9erD8k6YopXyeBvriDHwQulwGvz4HG6ae/t7mGfbDICCo7Pv1D4Sg8FJLFT83ogvR9YN5cOJyjp8ooQDakQPwr+4w542yt0UpVSdooC+OCIx4C5p1hUk3w771dvnBLbbrZXnSNk6OQVMliQgL5O+Xd2XT3mxe+2V98SvOfdOmmdJqZvVnpZT3aaAvSVAYjB5vC5Z9MdoGZ+dsUhUJ9GENT+teWZyz2kVzTd94PpyzxX2Xy+y9sPAD+3zvmvK3Rynl0zTQlyYiDq78zBYwm3QTbJxmJwFv1Lb8+/Tgit7piQs60qJhGA9PWk52Tt7pb855A/Jzocd1cDhdZ65SSrmlgd4T8f3gotft1fyGn8pe9qCwMgT6esEBvPanbuw4dIwXf1x76o3Du2DROOh2FXQaaZft0at6pVRRGug91fN66HO7fd52WMX2VYZAD9C7VRS3Dkrgi4Xb+WLhdrvw99fA5MNZj0CTTnbZntUVa5dSyid5NMOUcjjvJRvk255dsf2ERtlAb4zH3wwePKcd63Zn8fjXKzmydys3L/0U6XEtNGxl9xMSCXs10CulitIr+rLwD4B259kqlxUR2tBejecc9niTkEB/xo1J4vJecYTNf4P8AsOJMx+0b4pATGdN3Sil3NJA7w0llEEoSaC/H68Oa8CVgbP5/EQyt363hyPOG7QxnW3PmwItm6CUOp0Gem9wBnoPulgWJr/9A3+/AMLPeZTfNuxj9Pvz2ZeVY/P0udmQub2SG6uUqu000HtDWMn1bop1YDMs/wKSbuLSs/rw/nVJbNybxWXv/MHS3Fi7jt6QVUoVooHeG8qZumH2K+AfBAP/DMDZnWL4wjFD1bWTbb7/+2m/8vWSHRw6kluZLVZK1WLa68YbyhPoD2yGlROh/90QHnNycY/4hvz+6FDmbNrPgW+bE3JwLfdOXI6fQK+WDRmUGE2P+Ei6tYikQUhgJZ+IUqo20EDvDeUJ9CkfgfhB/3uK7i7In3M6xcDyHpx9YCPfjTyT6Wv38Ovavbw+bcPJ9do2qU/3FpF0bxHJWe2iaREVVtEzUUrVAhrovcHDCpYnnTgGSz+DDhdBeNPi14vphGz4iW5NQ+jWoj0PntuezGMnWLEjg2XbM1iWlsHMdXuZtHgH4cEB/PboEBrWK0cFTqVUraKB3lvKMjp29Td2IvLeN5e8XkxnMAWwbx007w5ARGgggxKjGZQYDYAxhiXbD3HZO/P4+I+tPHhOu/Kfg1KqVvDoZqyIDBeR9SKySUQec/P+gyKyRkRWiMh0EWnp8l6+iCxz/JtcmY2v1UIjPe9euWgcNG4HrQaVvF6TzvaxhEqWIkKvllGc1zmGT+ZuIeu4TliilK8rNdCLiD8wFjgf6ARcJSKdCq22FEgyxnQFJgGvuLx3zBjT3fFvRCW1u/YLi/Lsin7nMkhPgaSbSy+XEJUAASEedbG8e0hbDh/P4/MF2u9eKV/nyRV9H2CTMSbVGJMLTABGuq5gjJlpjHHOpD0fiKvcZvogT1M3KeMgMMzOY1sa/wCIbu9RoO8aF8mgxMZ8+PuWkmewUkrVep4E+lggzeX1Dsey4twM/OTyOkREUkRkvoiMcreBiNzmWCdl3759HjTJB0S1gYObYevc4tc5ngkrJ9kpDZ1zzZYmpovHk5DcPaQt+7NzmJiSVvrKSqlaq1IHTInItUAS8KrL4pbGmCTgauBNEWlTeDtjzPvGmCRjTFJ0dHRlNqnmGviArTz59a3F5+qXT4ATR6H3LZ7vt0knyN4DR/aXumrf1lH0atmQ92anciJfa+Qo5as8CfTpQAuX13GOZacRkbOBJ4ERxpgc53JjTLrjMRWYBfSoQHt9R3A4XDbOTgf4/X221LArY+xN2NheJ3vQeCTGcUPWg/SNiHDPkLakZxzj26VFfqVKKR/hSaBfBCSKSGsRCQJGA6f1nhGRHsB72CC/12V5QxEJdjxvDJwJaC1dp9ieMOxpWPs9LP749Pe2zoH96+1N2LIoQ6AHSG4fTcdmDXhn9mbyC0zpGyilap1SA70xJg+4B5gKrAUmGmNWi8jzIuLsRfMqUB/4X6FulB2BFBFZDswEXjbGaKB31f8eOzXhz4/DXpepAlPG2clEulxatv3VbwL1oj2ehEREuHtIG1L3HWHq6t1lO5ZSqlYQUzhl4GVJSUkmJSXF282oXll74N0zoV4TuHU6HD8Mb3SCvnfAeS+WfX+fjoCcLLhtpker5xcYznl9Nl39NvOG/7+Qm6aWPAJXKVXjiMhix/3QIrR6ZU0QHgOj3rFX4b88BUv+AwV5kHRT+fYX09l+OyjwrNukv59wR3IbWh/8HTm0FTb8XL7jKqVqJA30NUXiOTaNs+gDmPtPSEiGRkU6KHkmpjPkHYNDWz3eZFT3WPoGbgHAbJpevuMqpWokDfQ1ybCnoWlXyM0qW5fKwpo4Bi7vWeXxJkH+Qg//VACOrpvBpt1lrJWvlKqxNNDXJAHBMPpzOO8laHd++fcT3cGWNC7LZOEHUwnOO0x6VD/qmWwe//envPDDGg5rLRylaj0N9DVNZLydXMS/AoVFg8Js3RsPe94AkL4YgNiLn8Qg3B63jY/mbmHIq7P4ctF2CrTrpVK1lgZ6XxXTuWzzx6YvtjV14gcgsT05O3AV398zkNaN6/GXr1Yycuxc5m0+UHXtVUpVGQ30vqpJZzi4BXKPeLZ++mJo1t1+k2gzFNJT6BJl+N8d/fnn6O7sz87hqg/mc/1HC1mVnlmlTa8xCvLh371g2Xhvt0SpCtFA76tiOgMG9q4rfd28XNi1AuJ62ddthtkJTLbMRkQY2T2WmQ8n8+QFHVmxI4OL/j2Hu8cvIXVfdpWegtdl7YIDm2DLb95uiVIVooHeV8U4et54kqffswryc2xdHYC4JDvV4eYZJ1cJCfTn1sEJ/PboEO4b2paZ6/Zyzhu/8fjXK9mVeawKTqAGyHBU9dy/oeT1lKrhNND7qshWEFjPszy940bsyUDvHwgJZ8GmGUWKrTUICeTBc9sz+5EhXNevJZMWp3HWK7N48puV7Dh0FJ+S4ZiUZd+GokXnlKpFNND7Kj8/e1WfvqT0ddOX2Po4ES5FStsMgcztNnXhRnR4MM+O6MzMh5O5IimOiSlpJL86i8e+WsH2Az4S8J2BPjcLsrQOkKq9NND7ssRzYcciOLyz5PXSF9uredepCtsMs48u6Rt34hqG8eIlZzD7kSFc0zeer5emM+S1WTw0cTlb9nt4I7imynSZZnH/eu+1Q6kK0kDvyzqNAgysKWFO9uOZNgcdW6gWUlRr2xffw3IIzSNDeW5kF35/dAhj+rfix5U7Ofv12Tz+9Yram8PP2A4NHLNi7t/o3bYoVQEa6H1ZdDs7teDqb4pfZ+dSwNja+IW1GQpbf4e8nKLvFSOmQQhPX9yJ3x515vB3kPzqLF6aspZDR3LLfg7elLEdWvSG4AawT6/oVe2lgd7XdR4FafMhs5gZpJw3Ypu7mfirzTA7lWHagjIftkl4CM+O6MyMh5K5sGszPvg9lcGvzOTf0zdyJCevzPurdgUFkLnDjlRunKg9b1StpoHe13W6xD6u+c79++lL7ETlYVFF32s1EPwCSs3Tl6RFVBiv/6k7Ux8YTP82jXht2gYGvzKTsTM3kXmsBtfRyd4D+bmOQN9eA72q1TTQ+7rGbaHpGcWnb9IX237z7oQ0gBZ9Pc7Tl6RdTDjvX5/E13cN4Iy4CF6dup6BL8/g5Z/WsS/L89RQtXH2uIlwXNFn7bITwihVC2mgrws6XwI7FtpUhKvMdBvAnP3n3WkzBHavgOx9ldKUnvEN+eTGPvx430DOah/N+79t5sy/z+Cv364k7WAN6paZ6RgsFRkP0e3tc70hq2opDfR1QadR9rFw+qbwQCl3nN0sUz2bltBTnZtH8NbVPZn+UDKX9ojly0VpJP9jFo9/vZI9h49X6rHKJWObfYxsAY3b2eeavlG1lAb6uqBRGzuhSeH0Tfpi8Au0PXOK06wbhEZVKE9fktaN6/HyZV35/dGhXNs33o60fXUmf/95nXdz+BnbIawRBNWDhq3svQrtS69qKQ30dUXnS+zgqQyXQUDpi6FpFwgMKX47P3+bvtlctBxCZWoaEcJzI7sw/cFkzuvclHdmbWbwKzN5b/Zmjp/wbO7bSpWRZtM2YEtCRLXR1I2qtTTQ1xWdR9lHZ/qmIB92Lis6UMqdNsNsL5Sy1Lcvp/hGYfxzdA9+vG8gPeIj+b+f1pH86iwWbjlY5cc+Tcb200tCNE7UvvSq1tJAX1dEJdh68870zf4NtoZLSfl5p7bDbIpnzhueH+/44QrdwO3cPIJPbuzDhNv6ERbkz82fLGLNzmrq9WKMvRnrvKIHe0P20BbIr8FdQpUqhgb6uqTzJTZdc2ibZzdincKbwuBHYNUkWPdj6evnHoWPzoPX2sOEa2DTr3YAUjn0S2jEZ7f0pX5IAGM+Xlg9PXOO7IO84xDZ8tSyxu2gIA8Oplb98ZWqZBro65KT6ZtvbaAPbgCN2nq27aAHIeYM+OFBOHao5HWnPAx710L3q2H7fPjsMvhXd/uNoBxX+c0jQ/n0pj7knMhnzEcLOZBdxf3unfcxIl1TN9rzRtVeGujrkoatoHlPm75JX2zLHvh5+CfgHwgj37JXuz8/Ufx6Sz+DZZ/bbwAj34IH18LlH9k0yK/PwusdYcojZb6x2y4mnI9u6E16xjFu+mRR1ZZROBnoXVI3jRPto+bpVS2kgb6u6XyJLWS2e2XxI2KL07w7DPwzLB8PG6cVfX/PavjxYWg1CJIfs8sCgqDLZXDDD3D3Iuh4ESx83+a7yyipVRRvXd2TlemZ3Pn5Ek7kly8dVKqTo2JdruiDw6FBrPa8UbWSR4FeRIaLyHoR2SQij7l5/0ERWSMiK0Rkuoi0dHlvjIhsdPwbU5mNV+XQaaR9NAWe5ecLO+tRiO4I399vSxw75WTBxDG2bMJl42y3zMKi28FZf7HPt83z/JjHDp38BnBOpxheuuQMftuwj0cnraCgoAq6fGZsh5BIey6uGidqX3pVK5Ua6EXEHxgLnA90Aq4SkU6FVlsKJBljugKTgFcc20YBzwB9gT7AMyLSsPKar8qsYctTAb48gT4gGEaNtaUTfnnKLjPGBv6Dm22QD48pfvvG7SG0IWz/w7PjHT0Ir3eGlI9OLhrdJ56HzmnHN0vTeWTSCrIrO41TuMeNU+P29opepxVUtYwnV/R9gE3GmFRjTC4wARjpuoIxZqYxxtkdYj7gmK2B84BpxpiDxphDwDRgeOU0XZXbwD9Dt6ttb5ryiO0FA+6FJZ/C5pk2CK/6CoY8Ca0Hlbytnx/E9/f8in7LbDhxBFZ9fdrie4a25b6hbfl66Q6Gv/kb8zYfKN+5uJOxvZhAnwi52aXP2KVUDeNJoI8F0lxe73AsK87NwE9l2VZEbhORFBFJ2bevcopnqRJ0vBgueadi+0h+3PbY+fZO+PkxaHs2DHzQs23j+9ur/6w9pa+bOss+bv/DXt07iAgPntue/93enwA/4aoP5vPs5NUcy63gKFpjig/0J4ubac8bVbtU6s1YEbkWSAJeLct2xpj3jTFJxpik6OjoymySqiqBoTDybTtpdr1ouOR9z3vwtBxgH7d7cFWfOtv2ZzcFsGFqkbeTWkUx5f5B3DCgFZ/8sZUL/vU7i7dVYBTt0YN2shW3V/TaxVLVTp78z0wHXLofEOdYdhoRORt4EhhhjMkpy7aqlorvC9f8D8Z8D/Uaeb5ds24QGFZ6oD+0zfbO6XcnhDeD9e4Ha4UFBfDsiM6Mv7UvJ/IL+PWDx9n9UjfWvXcDq6a8x7bNa8n1tF7OyaqVbgJ9/RgIjqh9gd4Y28tK7y3UWQEerLMISBSR1tggPRq42nUFEekBvAcMN8bsdXlrKvCSyw3Yc4HHK9xqVXMknlP2bfwDbdfObaXckN0y2z4mDLHBdfmXcOJ4sUXYBrRpzM/39MHv9es5nBtEs51Tidj1DSyEnaYRqwM6s6XJ2Zx96c0kRNd3f0x3XSudRGpnzZu138PE6yDpZrjgH55/81I+o9TfuDEmD7gHG7TXAhONMatF5HkRGeFY7VWgPvA/EVkmIpMd2x4EXsB+WCwCnncsU3Vd/ADYs+r0LpqFpc6C+k1tbrz9hfamrDP4F6N+6s+E5WfR9PqP4NGtrLvkZ5af8QRZjXvQl5Xctutpbv/nRN6etcl9P3zXCUfciW5f+/rSOz9QU8bBd3dBfi2Ys1dVKk+u6DHGTAGmFFr2tMvzs0vY9iPgo+LeV3VUy/4275620P23goICm59vO8xeSbceBEHhttZOu/OK3++ST21Ov9VgIvz8iOjWH7r1t+8dTIV/9eDmplt47OcYfli+i1cu70qX2IhT22dst+mZ0Ej3+2+caEf+Hs+EkAj369Q06Sn2BnjbYTDjb5B7xHaDDQjydstUNdHvcMo74nrbyTyKS9/sXQNH90Prs+zrgGAbqDb8XHyBtIOpsOU36HGd+/REVAI0bMXoRpt499pe7MvOYeTYufzflLWneutkbIfIFhhjOH4inwPZORjX3HbjWjatYF4u7Fphu8QOfgTO+z9YOxkmXA0njnm7daqaeHRFr1SlC6pnb8oWd0PW2a0y4axTyzpcaAuy7VzivnzD0s9A/GwxteK0GQorJjL8iij6tzmL/5uylvd+S+X75TtpHB7MGwfXklYQzW1//ZlcR2qnW4tInrqwI0mtok71vNm33n0bNs+EfevsDeSaYM8qyM851db+d9mf/ff3w2eXw9UTbHkH5dP0il55T3x/W1zthJs5YrfMhkaJEBF3alniOSD+7ksl5+fB0s+h7TkQUcIwjzbD7KCnHYuICA3k5cu6Mv6WvrSNCScqLJDm7COocStuGtiaR4e359Hh7dmdeYzL353H3eOXkEYTW5vfTc8bs2c1BV9cZccV1JRBVe7KUfcaA5d9aD9k/zOy5PskFbVrBUy+t/SKp6pKaaBX3tNyAOTn2it0V3m5sHXu6VfzYEsntDoT1k+hiE3TIHs39Ly+5GO2HmQ/LFzmwB3QtjH/uakPn1zVjtCCowzo2YPHzu/AXcltuSu5LTMfTub+YYlMX7uHYW/MZV9wHCf2rAMg42guP6zYyVMT5pL2zmUcPmH/S83+7qOK1+HZPBMyd1RsHztSoF6Tor2Izrgcrvyv/SBY+lnFjlGcVV/BuHNhyX9g/c9VcwzlEQ30ynviHTdJC+fp0xfbHjYJyUW3aX+BTY0c2Hz68iX/sQGtpBu1YG+gxvWGzdOLvueuPDG2n/6fz2nHrIeHcFG3ZqRkNyZ90wpGjZ1Lzxemce/4xQxd9zSx7GXxgLfZFdSKoA0/cvtni8k6Xs4ZqTK22zr+X91Ssf7v6Sk2bSNS9L0OF0KDOPthUJkK8m1J6kk32fRcUDjsWFi5x1BlooFeeU9YFER3KJqnT51lc+2tBhbdpv0F9tH1qv7wLjtqtvvVto9+adoOs/PlHilUH6eYQO/UNCKE1//UnZ69+tKC3QRxgnuHJjJ3wDKGkIL/8JcYdt4omva7gr7+61i+biMjx85l096s0ttU2IL3wOTbn40ns3q5c+wQHNhUcvG62J5Fv1FVxLEMGH+lnWSm1w12MF1cLzsxvfIaDfTKu+L72y6WBS4jV7fMtvPbhropdNqwJcR0gXUugX75eBsUS0vbOLUZChjYMuv05Rml9KF3iGndFX8KmHhFU/7cOo3mS16DM66AvrcDIJ1G4UcBXww6QObRE4x8ay4/r9rtWdvAlnxe8h9bUrpxe/j1mfLNVZvuCOAlzTsQ2xMObT2tjlC57VsPHw6D1Jlw4etw8T9tF864Pnaugpzsih+jqu1eCXvXebsVlU4DvfKulgMg57DtHQI2yO1YVDQ/76r9BZA2316RFxTAkv9Cy4HQqI1nx2zew9ab3zTj9OUZ2yGovvsPGFfRjp43m36Fr26GJp1sUHOmR2I6Q1QCbfZP54f7BtI2Jpw7PlvMCz+sYea6vazbfZjMoydO77bpaunn9mcy4H445zl7Vb74E8/OzVX6YkDsrGLFcb5X0av6tEXwwTB7Y3fM99D75lPvtehjx0xU5jeHqjLpJtv11MfKRWj3SuVdJ/P082w+d9sfdhJud/l5pw4XwG+v2D71kS1sPZzkMlTW8PO3+988w/6HdgZoZ9VKd/lsV40c0wpOe8rmn6/8r+2y6CQCHUfAvLdoFniML2/rx7OTVzNuzhbGzTk1s1ZYkD/NIkJoHhlKcvsmXNy1GU3qB8L8t6FFX5vyMMZ+iM16GbpeWXQylJLsSLEjeUvapnl3+5i+1FYgLa9ZL9mfwa3TT+8pBadSRzsWQevB5T9GVcvafao31dY5pZfcrkX0il55V2QL2yPEORFJ6mwICIEW/Yrfpll3O63f+ik2xREcAZ1GFL++O22GQtZOe2PXKXO7+xo3hQXXtzcxTQFc+r77bxKdRtoPrPU/ERLoz8uXdWXBE8P46s4BvHV1D/56YUdG946nfdNw9h7O4YUf1tD3/6bzxtg3IWMbR3raNBAicO4LdvDY3Dc9Pz9j7I3Y0iaXCYmwH1zObpjlsX+T/dDsfXPRIA/2XkyjRHvVX5Ntm2sfxd+OsPYhekWvvC++v70Ba4x9bNG32MJlgA1+7c+3KQ5TYHPzgaFlO2abofZx8wxo0tE+z9he8geMqzPvs98M2hczj07zHhARb0eh9rgGgJgGIcQ0CAGKpoY27c1m8vKdDJ33N3aYxpwzKZhBK1O4d2giZ8T1tPcA5o21hclKGifgdGgrHD3g2Sxisb1sXt31201ZpIyzo5x7ljBTaFxv2Di1/MeoDlvn2m9oXa+wXU7Pf8V+SPkAvaJX3teyPxzZC2kLYO/qktM2Tu0vgLxjdtSnpzdhXUW2sKNcnf3pj2faf5EeXNGDvfHa+5bi3xexE7xsngHHD5e6u7ZN6vNgp2y6FaxB+t7B1f0TWLL9EJe/+weTl++EoU/ZD7WZL3rWPucVuicTwMf2hOw95RvklXvEfuB2HFHyFJItetsPnnJMCl9tts21pbeTbrbjO5ZP8HaLKo0GeuV98Y6JSGa9bB9LuhHr1GoQBDewaZxmXct33DZD7VXcieMe97gpk04jbcBwM2GKW/PfhqBwYofexlMXdWLqA4PpGhfBfV8s5Y2UHEyf22HZeNszpDTpiyEgFJp0Ln3dityQXTkJcjKhz60lrxfXxz7W1PTNkf02jdfyTGjaBWKTbPrGR27KaqBX3hfdHkKjbPogJMIG79IEBNmboKPeLv9x2wy13wq2zyu1D325xPW2E6as/a70dQ/vhNXfQM/rTlbFbFQ/mM9u6ctlPeP45/SNPLr3HExIxKlJ2UuyIwWad8f4+XPwSC7bDhwpft2mZ9jUS3oZA70xsOgD+2HivKlenCYdbY+mmtqf3pmfd47d6DXGBv60Bd5rUyXSQK+8T+RUoGg1yOa+PZGQbLsyllfLM23dms0zXCYcqcRA7+cHHS6Cjb/aFEdJFr5vUzOOvvhOwQH+/OOKrvxleAcmrclmnP+f7Afipl9PWy/z6AlSth5k4qI0/jFlJSfSl/L13qZ0ffYXer4wjbNencVNnyxiy3437QgMsV1Ey3pFv2OR/XbR55bS8+5+/jZFVFNHyG6da2c9a97Dvu58qc3XL/aNm7J6M1bVDC3726kCPcnPV5bg+hDfzwb6hGSb6qjXuHKP0WmkverdOA06j3K/Tu4RSPnYliRo2KrI2yLCncltSIiuxyMTDOf7f0foN3/h34njWL/vOBv3ZrMvK+fk+t39U3k48AQ763fm0o6xxDeqx9GcPN77LZVz35jNTWe25p6hbQkPcRlFHNsLVn9txyV4OgPVwg9s+uyMP3m2flwfO2I298jp3VEL88YN221zbX9/58jq4Pq2HtDyCTD8/4qfn6CW0Ct6VTN0vNj2tulwUfUet81QO1grbaFnfejLquUACGtse98UZ/kXcDwD+t1d4q7O69yU8XcM4q2A64k6sglZ+l+O5uaT3C6aJy7owEc3JDH7kWS+uthOKHLPtVfy3Mgu3DywNfcOS2TGw2dxSY9Y3v89lSH/mMXERWmnCq/F9rQ3ow+menZe2ftsyehuV9mg6Im43nYE885lxa9jDHw0HKY86tk+K8PRg3bkbstCJTd63WBTeyv/V31tqSIa6FXN0LAV3PwLNGhWvcd1drPcsdDzHjdl4edvr9Q3THVfjjknG+a/Y1MG8aV37ewSG8Ezjz5GTmw/nqr3Dd/e1JlXr+jGbYPbMLRDDC0b1cN/5xK3FSubhIfwyuXd+O7uM4mPCuPRr1Ywcuxc1u0+XPYbskv/Y280l9TzqLC43vaxpPTN1jl21PPqb6rvRuj2eYCxlVFdNe9uB/Et/qTW35TVQK/qtqZd7RU3VO6NWFedRtoa+C6lkTm8E6Y9A290siUOBj7o8beJkKAAgi96BTl6EH57tegK6YuLr1gJdI2L5Ks7B/Dmld3ZlXmcS8b+wY+7G9jUlSc3ZAvybaqp9eBT5SA8Ua+RneWrpJ43C961j0f2wt61nu+7IrbOtYP03I056HWD/cZX1hvVNYwGelW3+flBmyH2eVUF+taDbW2dtZPtRBxf3w5vngF//MtOlXjztLKP7G3WDXpcawPj/k2nlh87BAc2ljpQSkQY1SOWH+8bSMdm4dw9YSVpIYkYTwLahp/tJOq9S+lS6U5cH3sT190V8qFtdrSzM+fvnGWsqm2bY79tBAQXfa/L5fYm7eKPq6ctVUQDvVLO9I0n5Q/Kwz/QDvBa8SW8NwjWfm9THvcusV1EW/Qp336HPW2vwqe5dLf0pGKli5gGIUy4rT/X9I3nl4xYctOXkZF9tOSNFn1oS1A4S0aXRYve9mo9Y5ub/X4ACJz9DDRqWz2B/liG/fBteab790MaQJfLYNXXHg18q6k00CvV8WJ7deoM+FWh9822vPLZz8GDa+D8v0NU64rts34TGPyQvQrePNMu86RiZSFBAX68eMkZdExKJtjk8MBbX7BmZzFBzVnXpteN4F+OTnsn8/SFJjvJPWLrFnW82NbLSUi2+frylGcui+3zcZufd9XrBjsRzqpJVduWKqSBXqngcLjwH1Vb1yQuCe74HQY+ULld9frdZW9kT33CzpvrScXKYgwYdA4A7U5s4NJ35vLs5NV8+sdWZq3fy9b9RziRX2D7+/sFlq/sBNjBVYFhtpeTq+UTbK8f56TqCck2uFb27FeFbZsD/kGnPoDcie1lP6QXfgh5OcWv56njmfZ3VY20H71StVlAMJzzAky8DpZ8YitWtium0FppohIgJJI/J2az8XAjJqakcTT31IQw7f3S+THoQ6YHn81rH67j+Ik1HDuRz3HHv+j6wXwwJonOzSOKP4Z/gP224drzxhg7o1azbraLLdgRquJn0zctSxl1W5yN0+DX5+CCV4vfx9a5NpCXVBRPBAY/Av8bA9/cAZeN83ysgVPuUTtT2IoJ9htRv7vgPA/rFlUCDfRK1XYdL7Z9wKc9Y3v3eFKx0h0RaN6D0H3L+fiOPhhj2Jedw7YDR9m2/wh9fnuNnOwwfoi+hYTg+oQG+RMS6EdwgD8hgf5MXpbOVe/P5z8396V7i8jij9OiN/zxb44eySI4tD7+W2bB/vUw6t1TPYVCG9oup6mzYEgZ5hoA+8Ex/2345a92tPFXN8Odc4tOKJOTBbuWw8A/l77PzqMg43mY9jTUi7apt9J6SRUU2IFYyyfAmu8gN8veB2rcztYsGvaMLeVRDTTQK1XbidjRm+85JvXw8EasW7E9Yc6bcOIYEhhKk/AQmoSH0DtrBhxeDBe+zr97n+t202v6xnPNhwu49sMFfHxjb3q3cp8Ky2ueREBBHjf93wdsCevKf8PeICG0Mf6dL+G00JmQbNty/LDnqai8XPjxQVj6Xzv4ru8d8N9R8P0DcMUnpwfn7QvsAC53cxO7M+A+yN4L895y3B95uPh11/0IPz9ubzoHhUPnkdB1tL3pu2kajP+TLWPRoRw3tMvBo0AvIsOBfwL+wIfGmJcLvT8YeBPoCow2xkxyeS8fcJbb226MKWM/MqVUqZp1haQbYfW3tm5NecX2ssFv98pTvYGOH4apT9or7F43FLtpi6gwJt7en6s/nM/14xby4Zgkzmx7ekmJ1Tsz+dtUwxfAn2J2sSi0LW3T5vLv/FF8/+8FXNIjlhHdmhPTIISDjfrR1OQza9p3LAjsTdrBoxzNzScyNJCIsEAahgXRMCyQiLAgmkWE0D0qj8BJY+wkNoMfgeQnbIplyBMw/Xl7Fe2YGwCw+Xm/AM97PYnYNNmRfTDjBXtl36tQDf6s3TDlEduVtklnm+ZpfwEEhZ1ap81QO3ZjxYSaE+hFxB8YC5wD7AAWichkY8wal9W2AzcA7j7ijhljule8qUqpEl3wDxjy11P1WsrD2VsnfcmpADjrZVuv/qrxpRacaxoRwpe39ee6cQu48ZNFvHdtL4Z0aEJuXgFjZ25i7MxNRIaFcrReCy5tsotLI+ZhdvoTN+QeGq7L5dWp63l16nr8BAJNLsuCg0hd8AMfFDQirmEo9YID2LAni4yjJ8jOOXVDs52k8XHwP2gimazp+xrtBt5IqDOPfuYDdn7gKY/Y0cfOGcG2zrXnW1LdncL8/GDkWFtb/4cHbG2kDhc65i7+1KbP8nNsWmbAve5/F/6Bto5Oyse2e2c11NHx5Iq+D7DJGJMKICITgJHAyUBvjNnqeK+gCtqolPKEn78dfVoRDZrZ0srOiUv2rLaDsnrd4HHuPzo8mC9u7cd1Hy3gtv+m2Mqbi3ewbncWl/SI5emLOhH2c397UzJ/NtL5Ei5LTuKyZEg7eJQfVuziWG4eLaLCyFncl+tytzDm7vPx9zs9J56bV0DGsVyOb1lA88nPcYRQxuQ9yx+zmxHyxy+c1S6aYR1iaBAaQGDHFxi0cxRH/juGmWf+F39zghE7lyAD7i37z8g/EK74FP4zwk4mftGbtmvo9j9s9dWL/1n6RPVdr7Q/1zXflvgtqbJ4EuhjgTSX1zuAvmU4RoiIpAB5wMvGmG8LryAitwG3AcTHV9HoRKWUZ5r3tDVvjIEfH7L18Yc9XaZdNKwXxOe39OPGjxfytx/XEh0ezAfXJ3FOJ8csVHG9YeVE+7zvHSe3axEVxp3JLkEy5xx7AzR7d5E6SEEBfjSpFwTznoKwhkTc8iuf1mvKgtSDTF29m1/W7Gbq6j0n1z/f7wbeOfFPdn73DPMKOjEyKI+C+DPL18c8uD5c/T/46Dz49g478nnkWOh+jWelLJr3sDdll39ZYwJ9RbU0xqSLSAIwQ0RWGmM2u65gjHkfeB8gKSmpdlcPUqq2i+1hS0YveM8W/Brx73KNMYgIDeS/N/flu2U7ueCMpkSGufQwaeHotx6bVPLNY2fZ6i2zodvoou+v+NL2nLn0Q2jQnEBgYGJjBiY25rkRnUndn82JfIO/n+AnZ5E1cw93r/2SS6J2k3fAj6/2xXJlGcr1nKZeI7j+WzuVYtKN9gatp0TsVf2MF2zph4Yty9kIz3jyYZYOuI4Nj3Ms84gxJt3xmArMAnqUoX1KqermTNFMfcJeeXe/tty7qhccwNV9408P8mAHICWeB0P/WvIOYs5wzD42q+h7uUfsTdbYXrZMQSF+fkLbJuF0bNaAdjHhtG1Sn/BRryFRCcQeXMC2oLa88Esa6RnHyn1+RMRB8l/KFuSdujpq+qyYWP7je8iTQL8ISBSR1iISBIwGSiiufYqINBSRYMfzxsCZuOT2lVI1kHOWJQxc+FrZBwd5wj8Qrpl4qqBccfz87BzCqbOKFkL74y3I2gnnveR5G4Prw2Ufgl8A0d3Oo8AYHv96JcYbZYgj4213yxUTqrwMcqk/HWNMHnAPMBVYC0w0xqwWkedFZASAiPQWkR3AFcB7IrLasXlHIEVElgMzsTl6DfRK1WShDSFhiO2t0qybt1tj0zdZu2D/hlPLDu+CuW/aEtAe1PE/TWxPuHshDc59gr8M78BvG/YxafGOcjXtQHYOP6zYWf5vBV2vtGWqq7gMskc5emPMFGBKoWVPuzxfhE3pFN7uD+CMCrZRKVXdrv/W2y04xZmnT51l6/gAzPwbFOTB2c+Wb5+OXjHX9WvJjyt28cIPaxjcLpqYBiEebX4kJ49xc7bw/m+pJ7t5dmgazpAOTRjWoQk94hsW6SXkVqeRttvnigkQV84RzR7QomZKqZqtYSto2PpUnn7XCnsDtO/ttj5PBfj5CX+/vCs5eQU8+U3pKZwT+QV8Nn8bZ706i9enbWBg28Z8cWs/nrygIw3Dgvjgt1Quf3cevf42jQcmLCV1X3bJDQiNhPbnw6qv2L43k8XbDlXofIqjJRCUUjVfQjKsnGSrPv7ypE0vDSqhBEEZtG5cj4fPbc+LU9YyeflORnaPLbKOMYafVu3m1anr2bL/CH1aRfH+9b3oGW/r5/Rv04hbBydw+PgJft+wnxnr9vLLmt38unYvf7+sKxd2LWGKzG6jYc23/H3sW6yPGMgvDwzGz5NvA2WggV4pVfMlJNtZnma+CFt+g/NfrdQRpTcNbM2PK3fx7OTVdG8Ryf7sXNbtPszaXYdZtyuL9buzyMrJo11MfcaNSWJohyaIm/7yDUICubBrMy7s2oydGe24e/wS7h6/hJRtrXj8/I4EBZyeRDl8/ATPLGnMX00414TOI/7GP1d6kAcQr9xtLkFSUpJJSaniGtRKqdrl6EF4JQEw0CgR7ppXsVIPbmzck8WF/5pDbv6pAf7hIQF0bNqADs3CSWoVxYVnNPMs9+6Qm1fAyz+t46O5W+gRH8lbV/ckNtKWRF687SD3T1jGrszjfN3yG7ru/Q55eEO5P8BEZLExxu2gBA30Sqna4b2zYNcyuOpLaF/OmvulmLZmD+t3H6ZjswZ0aNaA5hEhbq/cy2rKyl08OmkFAf7Ca1d0Y2V6Jv+avpHYhqG8eWUPevmnwodD4eJ/FS2U5iEN9Eqp2m/FRDvjlCe14Gug1H3Z3PX5EtbtzgLg0h6xPDeyM+EhgbYf/VtJUD8GbpxSyp7c00CvlFI1wLHcfP49YyMdmzXg4m7NT39z9Tc24He5tFz7LinQ681YpZSqJqFB/jw6vIP7NztfUmXH1X70Sinl4zTQK6WUj9NAr5RSPk4DvVJK+TgN9Eop5eM00CullI/TQK+UUj5OA71SSvm4GjcyVkT2AdsqsIvGwP5Kak5touddt+h51y2enHdLY0y0uzdqXKCvKBFJKW4YsC/T865b9Lzrloqet6ZulFLKx2mgV0opH+eLgf59bzfAS/S86xY977qlQuftczl6pZRSp/PFK3qllFIuNNArpZSP85lALyLDRWS9iGwSkce83Z6qJCIficheEVnlsixKRKaJyEbHY0NvtrGyiUgLEZkpImtEZLWI3O9Y7uvnHSIiC0VkueO8n3Msby0iCxx/71+KSJC321oVRMRfRJaKyA+O13XlvLeKyEoRWSYiKY5l5f5b94lALyL+wFjgfKATcJWIdPJuq6rUJ0Dh2ZEfA6YbYxKB6Y7XviQPeMgY0wnoB9zt+B37+nnnAEONMd2A7sBwEekH/B14wxjTFjgE3Oy9Jlap+4G1Lq/rynkDDDHGdHfpP1/uv3WfCPRAH2CTMSbVGJMLTABGerlNVcYY8xtwsNDikcCnjuefAqOqs01VzRizyxizxPE8C/ufPxbfP29jjMl2vAx0/DPAUGCSY7nPnTeAiMQBFwIfOl4LdeC8S1Duv3VfCfSxQJrL6x2OZXVJjDFml+P5biDGm42pSiLSCugBLKAOnLcjfbEM2AtMAzYDGcaYPMcqvvr3/ibwKFDgeN2IunHeYD/MfxGRxSJym2NZuf/WdXJwH2SMMSLik/1mRaQ+8BXwgDHmsL3Is3z1vI0x+UB3EYkEvgGKmV3ad4jIRcBeY8xiEUn2cnO8YaAxJl1EmgDTRGSd65tl/Vv3lSv6dKCFy+s4x7K6ZI+INANwPO71cnsqnYgEYoP858aYrx2Lff68nYwxGcBMoD8QKSLOCzVf/Hs/ExghIluxqdihwD/x/fMGwBiT7njci/1w70MF/tZ9JdAvAhIdd+SDgNHAZC+3qbpNBsY4no8BvvNiWyqdIz87DlhrjHnd5S1fP+9ox5U8IhIKnIO9PzETuNyxms+dtzHmcWNMnDGmFfb/8wxjzDX4+HkDiEg9EQl3PgfOBVZRgb91nxkZKyIXYHN6/sBHxpgXvduiqiMiXwDJ2NKle4BngG+BiUA8tszzn4wxhW/Y1loiMhD4HVjJqZztE9g8vS+fd1fsjTd/7IXZRGPM8yKSgL3SjQKWAtcaY3K819Kq40jdPGyMuagunLfjHL9xvAwAxhtjXhSRRpTzb91nAr1SSin3fCV1o5RSqhga6JVSysdpoFdKKR+ngV4ppXycBnqllPJxGuiVUsrHaaBXSikf9/8nXZUWINlGjQAAAABJRU5ErkJggg==)
#

# # 3.2. VGG16 모델
# vgg16 모델을 이용한 테스트

# ## 3.2.1. 데이터 전처리 (VGG16)

if MODEL_TYPE == ModelType.VGG16 or MODEL_TYPE == ModelType.ALL:
    # ------------------------
    # 훈련용 데이터 전처리
    # ------------------------
    train_dategen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
    )

    # batch_size = 128
    batch_size = 20

    train_generator = train_dategen.flow_from_directory(
        train_dir,
        target_size = (IMAGE_WIDTH,IMAGE_HEIGHT),
        batch_size = batch_size,
        class_mode='binary')


    # ------------------------
    # 검증용 데이터 전처리
    # ------------------------
    valid_datagen = ImageDataGenerator(rescale=1./255)
    valid_generator = valid_datagen.flow_from_directory(
        valid_dir,
        target_size = (IMAGE_WIDTH,IMAGE_HEIGHT),
        batch_size = batch_size,
        class_mode='binary')

# ## 3.2.2. 모델 생성 (VGG16)

# VGG Model
if MODEL_TYPE == ModelType.VGG16 or MODEL_TYPE == ModelType.ALL:
    conv_base = VGG16(weights='imagenet',
                     include_top=False,
                     input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3))
    # conv_base.summary()
    conv_base.trainable = False

    model_vgg = models.Sequential()
    model_vgg.add(conv_base)
    model_vgg.add(layers.Flatten())
    model_vgg.add(layers.Dense(256, activation='relu'))
    model_vgg.add(layers.Dense(1, activation='sigmoid'))
    model_vgg.summary()

# ## 3.2.3. 모델 학습 (VGG16)

if MODEL_TYPE == ModelType.VGG16 or MODEL_TYPE == ModelType.ALL:
    # 컴파일
    model_vgg.compile(loss='binary_crossentropy',
             optimizer=optimizers.RMSprop(lr=2e-5),
             metrics=['acc'])
    
    # callback 관련
    callbacks_list = [
        keras.callbacks.EarlyStopping(
            monitor='acc',
            patience=1),
        keras.callbacks.ModelCheckpoint(
            filepath='cats_vs_dogs_vgg16.h5',
            monitor='loss',
            svae_best_only=True)
    ]

    # 학습 시작
    history = model_vgg.fit(
        train_generator,
        steps_per_epoch=15,
        validation_data=valid_generator,
        validation_steps=50,    
        epochs = 10,
        callbacks = callbacks_list)

# # 3.3. DenseNet201

if MODEL_TYPE == ModelType.DENSENET201 or MODEL_TYPE == ModelType.ALL:
    # ------------------------
    # 훈련용 데이터 전처리
    # ------------------------
    train_dategen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
    )

    # batch_size = 128
    batch_size = 20

    train_generator = train_dategen.flow_from_directory(
        train_dir,
        target_size = (IMAGE_WIDTH,IMAGE_HEIGHT),
        batch_size = batch_size,
        class_mode='binary')


    # ------------------------
    # 검증용 데이터 전처리
    # ------------------------
    valid_datagen = ImageDataGenerator(rescale=1./255)
    valid_generator = valid_datagen.flow_from_directory(
        valid_dir,
        target_size = (IMAGE_WIDTH,IMAGE_HEIGHT),
        batch_size = batch_size,
        class_mode='binary')

    # ------------------------
    # 데이터 형태 확인
    # ------------------------
    for data_batch, labels_batch in train_generator:
        print('Data batch shape:', data_batch.shape)
        print('Labels batch shape:', labels_batch.shape)
        break;

if MODEL_TYPE == ModelType.DENSENET201 or MODEL_TYPE == ModelType.ALL:
    conv_base = DenseNet201(
                    weights='imagenet',
                    include_top=False,
                    input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3),
                    pooling='avg')

    conv_base.trainable = False

    model_densenet201 = models.Sequential()
    model_densenet201.add(conv_base)
    # model_densenet201.add(layers.Flatten())
    model_densenet201.add(layers.Dense(128, activation='relu'))
    model_densenet201.add(layers.Dense(128, activation='relu'))
    model_densenet201.add(layers.Dense(1, activation='sigmoid'))
    model_densenet201.summary()
    
    
    # model_densenet201.compile(
    #    loss='binary_crossentropy',
    #    optimizer='adam',
    #    metrics=['accuracy']
    #)
    model_densenet201.compile(loss='binary_crossentropy',
             optimizer=keras.optimizers.Adamax(lr=0.001),
             metrics=['acc'])

# ```
# Model: "sequential_3"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# densenet201 (Functional)     (None, 1920)              18321984  
# _________________________________________________________________
# dense_9 (Dense)              (None, 128)               245888    
# _________________________________________________________________
# dense_10 (Dense)             (None, 128)               16512     
# _________________________________________________________________
# dense_11 (Dense)             (None, 1)                 129       
# =================================================================
# Total params: 18,584,513
# Trainable params: 262,529
# Non-trainable params: 18,321,984
# _________________________________________________________________
# ```

if MODEL_TYPE == ModelType.DENSENET201 or MODEL_TYPE == ModelType.ALL:
    history = model_densenet201.fit(
        train_generator,
        validation_data=valid_generator,
        epochs = 15,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True)]
    )

# ```
# Epoch 1/15
# 748/748 [==============================] - 136s 169ms/step - loss: 0.1703 - acc: 0.9260 - val_loss: 0.0596 - val_acc: 0.9749
# Epoch 2/15
# 748/748 [==============================] - 123s 165ms/step - loss: 0.1125 - acc: 0.9520 - val_loss: 0.0512 - val_acc: 0.9791
# Epoch 3/15
# 748/748 [==============================] - 123s 164ms/step - loss: 0.1106 - acc: 0.9540 - val_loss: 0.0523 - val_acc: 0.9794
# Epoch 4/15
# 748/748 [==============================] - 123s 164ms/step - loss: 0.1039 - acc: 0.9561 - val_loss: 0.0673 - val_acc: 0.9725
# Epoch 5/15
# 748/748 [==============================] - 123s 165ms/step - loss: 0.1000 - acc: 0.9600 - val_loss: 0.0489 - val_acc: 0.9808
# Epoch 6/15
# 748/748 [==============================] - 123s 164ms/step - loss: 0.0920 - acc: 0.9624 - val_loss: 0.0482 - val_acc: 0.9802
# Epoch 7/15
# 748/748 [==============================] - 124s 165ms/step - loss: 0.0969 - acc: 0.9601 - val_loss: 0.0490 - val_acc: 0.9791
# Epoch 8/15
# 748/748 [==============================] - 123s 165ms/step - loss: 0.0928 - acc: 0.9638 - val_loss: 0.0459 - val_acc: 0.9824
# Epoch 9/15
# 748/748 [==============================] - 124s 165ms/step - loss: 0.0870 - acc: 0.9628 - val_loss: 0.0571 - val_acc: 0.9767
# Epoch 10/15
# 748/748 [==============================] - 126s 169ms/step - loss: 0.0837 - acc: 0.9661 - val_loss: 0.0535 - val_acc: 0.9781
# Epoch 11/15
# 748/748 [==============================] - 124s 165ms/step - loss: 0.0795 - acc: 0.9681 - val_loss: 0.0469 - val_acc: 0.9824
# Epoch 12/15
# 748/748 [==============================] - 124s 166ms/step - loss: 0.0796 - acc: 0.9665 - val_loss: 0.0475 - val_acc: 0.9824
# Epoch 13/15
# 748/748 [==============================] - 124s 165ms/step - loss: 0.0843 - acc: 0.9661 - val_loss: 0.0473 - val_acc: 0.9834
# Epoch 14/15
# 748/748 [==============================] - 128s 170ms/step - loss: 0.0780 - acc: 0.9721 - val_loss: 0.0516 - val_acc: 0.9812
# Epoch 15/15
# 748/748 [==============================] - 124s 165ms/step - loss: 0.0754 - acc: 0.9697 - val_loss: 0.0484 - val_acc: 0.9818
# ```


