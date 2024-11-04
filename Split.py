import cv2
import os
import numpy as np
import argparse

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Select Case")
parser.add_argument('--case',
                    type=int,
                    default=1,
                    help='1: Split image, 2: Combine image')
parser.add_argument(
    '--dataset',
    type=str,
    default='Shuguang_Village',
    help=
    'Shuguang_Village  Sardinia    River   Gloucester  Gloucester_ California Texas wuhan   Yellow_River'
)
args = parser.parse_args()

args.case = 2
args.dataset = 'Shuguang_Village'

size_dict = {
    'Shuguang_Village': [921, 593],
    'Sardinia': [412, 300],
    'River': [465, 827],
    'Gloucester': [1318, 2359],
    'Gloucester_': [554, 990],
    'California': [500, 875],
    'Texas': [808, 1534],
    'wuhan': [503,495],
    'Yellow_River': [291,343]
}

root_path = os.path.split(os.path.abspath(__file__))[0]

if args.dataset == 'Gloucester':
    IMAGE_SIZE = 128
    slide_step = 32
elif args.dataset == 'Texas':
    IMAGE_SIZE = 64
    slide_step = 16
else:
    IMAGE_SIZE = 32
    slide_step = 8

win_w = IMAGE_SIZE
win_h = IMAGE_SIZE
pic_h = size_dict[args.dataset][1]
pic_w = size_dict[args.dataset][0]


def make_dir(path):
    image_dir = path
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)


def Split_image(path, target_path):
    print(path)
    img = cv2.imread(path, 0)
    img = cv2.imread(path)
    #img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    num = 0
    img = cv2.transpose(img)
    for j in range(pic_h // win_w + 1):
        for i in range(pic_w // win_h + 1):
            if i != pic_w // win_h and j != pic_h // win_w:
                buff = img[i * win_h:(i + 1) * win_h,
                           j * win_w:(j + 1) * win_w]
            elif i != pic_w // win_h and j == pic_h // win_w:
                buff = img[i * win_h:(i + 1) * win_h, pic_h - win_h:pic_h]
            elif i == pic_w // win_h and j != pic_h // win_w:
                buff = img[pic_w - win_w:pic_w, j * win_w:(j + 1) * win_w]
            elif i == pic_w // win_h and j == pic_h // win_w:
                buff = img[pic_w - win_w:pic_w, pic_h - win_h:pic_h]
            buff = cv2.transpose(buff)
            if target_path == 1:
                make_dir(save_path + r'/testB/testB/')
                cv2.imwrite((save_path + r'/testB/testB/' +
                             'Image1_{:03d}.png'.format(num)), buff)
            else:
                make_dir(save_path + r'/testA/testA/')
                cv2.imwrite((save_path + r'/testA/testA/' +
                             'Image2_{:03d}.png'.format(num)), buff)
            num += 1
    print(f'The test images shape is : {num}.')


def Combine_image(path, save_path):
    img = np.zeros((pic_w, pic_h, 3))
    num = 0
    for j in range(pic_h // win_w + 1):  #0-9
        for i in range(pic_w // win_h + 1):  #0-5
            buff = cv2.imread(os.path.join(path, f'Image_style_{num:03d}.png'))
            buff = cv2.transpose(buff)
            num += 1
            if i != pic_w // win_h and j != pic_h // win_w:
                img[i * win_h:(i + 1) * win_h,
                    j * win_w:(j + 1) * win_w] = buff
            elif i != pic_w // win_h and j == pic_h // win_w:
                img[i * win_h:(i + 1) * win_h, pic_h - win_h:pic_h] = buff
            elif i == pic_w // win_h and j != pic_h // win_w:
                img[pic_w - win_w:pic_w, j * win_w:(j + 1) * win_w] = buff
            elif i == pic_w // win_h and j == pic_h // win_w:
                img[pic_w - win_w:pic_w, pic_h - win_h:pic_h] = buff

    for i in range(1, pic_w // win_h + 1):  #1-5
        for j in range(1, pic_h):
            for z in range(2):
                if i != pic_w // win_h:
                    window = img[i * win_w - 1 + z:i * win_w + 1 + z,
                                 j - 1:j + 1]
                    mid = np.median(window.reshape(-1, window.shape[2]),
                                    axis=0)
                    img[i * win_w + z, j] = mid
                else:
                    window = img[pic_w - win_w - 1 + z:pic_w - win_w + 1 + z,
                                 j - 1:j + 1]
                    mid = np.median(window.reshape(-1, window.shape[2]),
                                    axis=0)
                    img[pic_w - win_w + z, j] = mid

    for j in range(1, pic_h // win_w + 1):  #1-9
        for i in range(1, pic_w):
            for z in range(2):
                if j != pic_h // win_w:
                    window = img[i - 1:i + 1,
                                 j * win_w - 1 + z:j * win_w + 1 + z]
                    mid = np.median(window.reshape(-1, window.shape[2]),
                                    axis=0)
                    img[i, j * win_h + z] = mid
                else:
                    window = img[i - 1:i + 1,
                                 pic_h - win_h - 1 + z:pic_h - win_h + 1 + z]
                    mid = np.median(window.reshape(-1, window.shape[2]),
                                    axis=0)
                    img[i, pic_h - win_h + z] = mid
    img = cv2.transpose(img)

    cv2.imwrite(save_path, img)


def Slide_split_image(path, target_path):
    img = cv2.imread(path, 0)
    img = cv2.imread(path)
    #img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    num = 0
    img = cv2.transpose(img)
    for j in range((pic_h - win_h) // slide_step + 1):
        for i in range((pic_w - win_w) // slide_step + 1):
            if i != pic_w // win_h and j != pic_h // win_w:
                buff = img[i * slide_step:i * slide_step + win_h,
                           j * slide_step:j * slide_step + win_w]
            elif i != pic_w // win_h and j == pic_h // win_w:
                buff = img[i * slide_step:i * slide_step + win_h,
                           pic_h - win_h:pic_h]
            elif i == pic_w // win_h and j != pic_h // win_w:
                buff = img[pic_w - win_w:pic_w,
                           j * slide_step:j * slide_step + win_w]
            elif i == pic_w // win_h and j == pic_h // win_w:
                buff = img[pic_w - win_w:pic_w, pic_h - win_h:pic_h]
            buff = cv2.transpose(buff)
            if target_path == 1:
                make_dir(save_path + r'/trainB/trainB/')
                cv2.imwrite((save_path + r'/trainB/trainB/' +
                             'Image1_{:03d}.png'.format(num)), buff)
            else:
                make_dir(save_path + r'/trainA/trainA/')
                cv2.imwrite((save_path + r'/trainA/trainA/' +
                             'Image2_{:03d}.png'.format(num)), buff)
            num += 1
    print(f'The train images shape is : {num}.')


def MaxMinNormalization(x):
    return (x - min(x)) / (max(x) - min(x))


def select_training_samples(trainA_path, trainB_path, select_threshold=0.7):
    traina = os.listdir(trainA_path)
    trainb = os.listdir(trainB_path)
    variance_a = []
    variance_b = []

    for namea, nameb in zip(traina, trainb):
        imga = cv2.imread(save_path + r'/trainA/trainA/' + namea)
        imgb = cv2.imread(save_path + r'/trainB/trainB/' + nameb)
        variance_a.append(np.mean(np.var(imga.reshape(-1, 3), axis=0)))
        variance_b.append(np.mean(np.var(imgb.reshape(-1, 3), axis=0)))
    variance = MaxMinNormalization(variance_a) + MaxMinNormalization(
        variance_b)
    threshold = np.sort(variance)[int(variance.size * select_threshold)]
    num = 0
    del_num = 0
    for namea, nameb in zip(traina, trainb):
        if variance[num] > threshold:
            os.remove(save_path + r'/trainA/trainA/' + namea)
            os.remove(save_path + r'/trainB/trainB/' + nameb)
            del_num += 1
        num += 1
    print(f'The selected train images shape is : {num-del_num}.')


def find(path, save_path):
    pic_list = os.listdir(path)
    for name in pic_list:
        label = name.split('.')[0].split('_')[2]
        AB = name.split('.')[0].split('_')[3]
        if label == 'fake':
            if AB == 'A':
                print(path + '/' + name)
                img = cv2.imread(path + '/' + name, 0)
                cv2.imwrite(save_path + '/fake_A/' + name, img)
            else:
                img = cv2.imread(path + '/' + name, 0)
                cv2.imwrite(save_path + '/fake_B/' + name, img)


if __name__ == '__main__':
    if args.case == 1:
        dataset_path = 'RS_data\Heterogeneous_dataset'
        cur_dataset_path = os.path.join(dataset_path, args.dataset)
        data_file = [
            f for f in os.listdir(cur_dataset_path)
            if os.path.isfile(os.path.join(cur_dataset_path, f))
        ]
        save_path = os.path.join(root_path, 'split_result', args.dataset)
        print('Start Split Images.')
        image1_path = os.path.join(cur_dataset_path, data_file[0])
        image2_path = os.path.join(cur_dataset_path, data_file[1])
        #test
        Split_image(image1_path, 1)
        Split_image(image2_path, 2)
        #train
        Slide_split_image(image1_path, 1)
        Slide_split_image(image2_path, 2)
        #select
        select_training_samples(save_path + r'/trainA/trainA/',
                                save_path + r'/trainB/trainB/')  

        print('Images Spliting Process is over.')

    if args.case == 2:
        img_path = os.path.join(root_path, 'cae_result', args.dataset,
                                'test_img')
        print('Start to combine image.')
        Combine_image(img_path,
                      save_path=os.path.join(root_path, 'cae_result',
                                             args.dataset, 'gen_image.png'))
        print('Process over.')

