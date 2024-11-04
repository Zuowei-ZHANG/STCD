import cv2
import os
import numpy as np
import matplotlib as plt

root_path = os.path.split(os.path.abspath(__file__))[0]
save_path = os.path.join(root_path, 'Result_Img')

def find(path):
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
    find(root_path + '/images')