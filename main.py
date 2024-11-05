from tkinter import W
from tkinter.messagebox import showerror
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import sys
import time
import argparse
from imageio import imread, imsave
from PIL import Image
from skimage.segmentation import slic, mark_boundaries
from sklearn.cluster import KMeans
from skimage.future import graph
from networkx.linalg import adj_matrix
import scipy
from scipy.spatial.distance import cdist, pdist
from sklearn.svm import SVC
from tqdm import tqdm

from fcm import FCM
from scipy.io import loadmat

import CDTool
import sklearn.cluster as sc
from sklearn.cluster import MeanShift
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

parser = argparse.ArgumentParser(description="cd Case")
parser.add_argument('--root_path',
                    type=str,
                    default=os.path.abspath(
                        os.path.dirname(os.path.abspath(__file__)) +
                        os.path.sep + "."),
                    help='project root path')
parser.add_argument(
    '--dataset',
    type=str,
    default='Gloucester',
    help=
    'Shuguang_Village  Sardinia  Gloucester  Texas'
)
parser.add_argument('--dataset_path',
                    type=str,
                    default='RS_data\Heterogeneous_dataset',
                    help='dataset path')
args = parser.parse_args()


def get_config():
    # load parameters
    # 0:diff_threshold_c, 1:diff_threshold_uc, 2:win_size_c, 3:th_c, 4:win_size_uc, 5:th_uc, 6:Ns, 7:assign_th
    parameters_dict = {
        'Shuguang_Village': [80, 10, 20, 0.3, 20, 0.3, 1500, 4],
        'Sardinia': [60, 20, 20, 0.33, 20, 0.33, 1000, 4],
        'Gloucester': [60, 20, 100, 0.65, 20, 0.3, 2000, 4],
        'Texas': [35, 14, 50, 0.3, 20, 0.3, 2000, 4],  #0.3
    }
    return parameters_dict

# gray image to rgb image
def gray2rgb(data):
    data = np.expand_dims(data, axis=2)
    data = np.concatenate((np.concatenate((data, data), axis=2), data), axis=2)
    return data


def abc2rgb(a, b, c):
    a = np.expand_dims(a, axis=2)
    b = np.expand_dims(b, axis=2)
    c = np.expand_dims(c, axis=2)
    data = np.concatenate((np.concatenate((a, b), axis=2), c), axis=2)
    return data


def make_dir(path):
    image_dir = path
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)


def median(img, n):
    blur = cv2.medianBlur(img, n)
    return blur

# classify method for getting change map without uncertain regions
def classify_get_change_map(image_t1, image_t2, label0, label1):
    label0 = label0.reshape(-1, 1).squeeze()
    label1 = label1.reshape(-1, 1).squeeze()
    idx0 = np.where(label0 == 1)
    idx1 = np.where(label1 == 1)
    data = np.hstack((image_t1.reshape(-1, b), image_t2.reshape(-1, b)))
    train_data = np.vstack((data[idx0], data[idx1]))
    train_label = np.zeros(train_data.shape[0])
    train_label[idx0[0].size:] = 1

    #RandomForestClassifier
    rfc = RandomForestClassifier(random_state=0)  
    rfc.fit(train_data, train_label)  
    change_map1 = rfc.predict(data)  
    #knn
    knn = KNeighborsClassifier(n_neighbors=7, algorithm='auto')  # knn
    knn.fit(train_data, train_label)  
    change_map2 = knn.predict(data)
    #svm
    clf = SVC(kernel='poly')  #'linear', 'poly', 'rbf'
    clf.fit(train_data, train_label)
    change_map3 = clf.predict(data)

    return change_map1.reshape(h, w).astype('uint8'), change_map2.reshape(
        h, w).astype('uint8'), change_map3.reshape(h, w).astype('uint8')

# classify method for getting change map including uncertain regions
def classify_get_change_map_3(image_t1, image_t2, label0, label1, label2):
    label0 = label0.reshape(-1, 1).squeeze()
    label1 = label1.reshape(-1, 1).squeeze()
    label2 = label2.reshape(-1, 1).squeeze()
    idx0 = np.where(label0 == 1)
    idx1 = np.where(label1 == 1)
    idx2 = np.where(label2 == 1)
    data = np.hstack((image_t1.reshape(-1, b), image_t2.reshape(-1, b)))
    train_data = np.vstack((np.vstack((data[idx0], data[idx1])), data[idx2]))
    train_label = np.zeros(train_data.shape[0])
    train_label[idx0[0].size:idx0[0].size + idx1[0].size] = 1
    train_label[idx0[0].size + idx1[0].size:] = 2

    #RandomForestClassifier
    rfc = RandomForestClassifier(random_state=0)
    rfc.fit(train_data, train_label)
    change_map1 = rfc.predict(data)
    #knn
    knn = KNeighborsClassifier(n_neighbors=7, algorithm='auto')  # knn
    knn.fit(train_data, train_label)
    change_map2 = knn.predict(data)
    #svm
    clf = SVC(kernel='poly')
    clf.fit(train_data, train_label)
    change_map3 = clf.predict(data)

    return change_map1.reshape(h, w).astype('uint8'), change_map2.reshape(
        h, w).astype('uint8'), change_map3.reshape(h, w).astype('uint8')

# method for getting change map without uncertain regions
def get_change_map(image_t1, image_t2, label0, label1, case='svm'):
    label0 = label0.reshape(-1, 1).squeeze()
    label1 = label1.reshape(-1, 1).squeeze()
    idx0 = np.where(label0 == 1)
    idx1 = np.where(label1 == 1)
    data = np.hstack((image_t1.reshape(-1, b), image_t2.reshape(-1, b)))
    train_data = np.vstack((data[idx0], data[idx1]))
    train_label = np.zeros(train_data.shape[0])
    train_label[idx0[0].size:] = 1
    if case == 'rf':
        #RandomForestClassifier
        rfc = RandomForestClassifier(random_state=0)  
        rfc.fit(train_data, train_label)  
        change_map = rfc.predict(data)  
    elif case == 'knn':
        #knn
        knn = KNeighborsClassifier(n_neighbors=7, algorithm='auto')  # knn
        knn.fit(train_data, train_label)  
        change_map = knn.predict(data)
    elif case == 'svm':
        #svm
        clf = SVC(kernel='poly')
        clf.fit(train_data, train_label)
        change_map = clf.predict(data)

    return change_map.reshape(h, w).astype('uint8')

# method for getting change map including uncertain regions
def get_change_map_3(image_t1, image_t2, label0, label1, label2, case='svm'):
    label0 = label0.reshape(-1, 1).squeeze()
    label1 = label1.reshape(-1, 1).squeeze()
    label2 = label2.reshape(-1, 1).squeeze()
    idx0 = np.where(label0 == 1)
    idx1 = np.where(label1 == 1)
    idx2 = np.where(label2 == 1)
    data = np.hstack((image_t1.reshape(-1, b), image_t2.reshape(-1, b)))
    train_data = np.vstack((np.vstack((data[idx0], data[idx1])), data[idx2]))
    train_label = np.zeros(train_data.shape[0])
    train_label[idx0[0].size:idx0[0].size + idx1[0].size] = 1
    train_label[idx0[0].size + idx1[0].size:] = 2
    if case == 'rf':
        #RandomForestClassifier
        rfc = RandomForestClassifier(random_state=0)
        rfc.fit(train_data, train_label)
        change_map = rfc.predict(data)
    elif case == 'knn':
        #knn
        knn = KNeighborsClassifier(n_neighbors=7, algorithm='auto')  # knn
        knn.fit(train_data, train_label)
        change_map = knn.predict(data)
    elif case == 'svm':
        #svm
        clf = SVC(kernel='poly')
        clf.fit(train_data, train_label)
        change_map = clf.predict(data)

    return change_map.reshape(h, w).astype('uint8')


def show_img(img):
    cv2.imshow('img', img.astype('uint8'))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def retain_same_row(mat_a, mat_b):
    aset = set([tuple(x) for x in mat_a])
    bset = set([tuple(x) for x in mat_b])
    return np.array([x for x in aset & bset])


def cal_max_score(source_data, c_uc_data):
    score_matrix = np.mean(np.exp(-cdist(source_data, c_uc_data)), axis=1)
    return np.argmax(score_matrix)


def iterative_superpixel(source_data,
                         change_pixel_pair,
                         unchange_pixel_pair,
                         segments_map,
                         super_center,
                         adj_matix,
                         assign_th=2):
    uncertain_pixel_pair = np.zeros_like(change_pixel_pair)
    traverse_mark = np.zeros_like(change_pixel_pair)
    traverse_mark[np.where((change_pixel_pair + unchange_pixel_pair) != 0)] = 1
    unlabel_size = np.argwhere(traverse_mark == 0).shape[0]
    super_label = np.array(range(super_center.shape[0]))
    assign_rate = 0  #  %
    epoch = 0
    superpixel_mark = np.zeros((super_center.shape[0], 2))
    while assign_rate < assign_th:
        init_time = time.time()
        #unlabel_data = source_data[np.where(traverse_mark == 0)]
        change_data = source_data[np.where(change_pixel_pair == 255)]
        unchange_data = source_data[np.where(unchange_pixel_pair == 255)]
        super_idx = cal_max_score(super_center, change_data)
        max_super_change_center_idx = super_label[
            super_idx]  

        adj_type = superpixel_mark[np.argwhere(adj_matix[
            max_super_change_center_idx, :] == 1).squeeze()]  

        superpixel_unlabel_data_idx = tuple(
            retain_same_row(
                np.argwhere(segments_map == max_super_change_center_idx),
                np.argwhere(traverse_mark == 0)).T)

        if superpixel_unlabel_data_idx:
            superpixel_unlabel_data = source_data[superpixel_unlabel_data_idx]

            d_c = np.mean(np.exp(-cdist(superpixel_unlabel_data, change_data)),
                          axis=1)
            d_uc = np.mean(
                np.exp(-cdist(superpixel_unlabel_data, unchange_data)), axis=1)
            diff_d = np.abs(d_c - d_uc)

            if np.sum(diff_d[d_c > d_uc]) > np.sum(
                    diff_d[d_c <= d_uc]) and np.sum(
                        (d_c > d_uc).astype(int)) > d_c.shape[0] - np.sum(
                            (d_c > d_uc).astype(int)) and (
                                (np.array([1, 0]) not in adj_type)
                                or np.all(adj_type == 0)):
                change_pixel_pair[superpixel_unlabel_data_idx] = 255
                superpixel_mark[max_super_change_center_idx, :] = np.array(
                    [1, 1])
            else:
                uncertain_pixel_pair[superpixel_unlabel_data_idx] = 255
                superpixel_mark[max_super_change_center_idx, :] = np.array(
                    [1, 2])
            traverse_mark[superpixel_unlabel_data_idx] = 1
        else:
            max_super_change_pixel = change_pixel_pair[np.where(
                segments_map == max_super_change_center_idx)]
            max_super_unchange_pixel = unchange_pixel_pair[np.where(
                segments_map == max_super_change_center_idx)]
            if 0 not in max_super_change_pixel and 255 not in max_super_unchange_pixel:
                superpixel_mark[max_super_change_center_idx, :] = np.array(
                    [1, 1])
            elif 0 not in max_super_unchange_pixel and 255 not in max_super_change_pixel:
                superpixel_mark[max_super_change_center_idx, :] = np.array(
                    [1, 0])
            else:
                superpixel_mark[max_super_change_center_idx, :] = np.array(
                    [1, 2])
    
        super_label = np.delete(super_label, super_idx, axis=0)
        super_center = np.delete(super_center, super_idx, axis=0)

        ##############################################################################
        ##############################################################################
        ##############################################################################
        change_data = source_data[np.where(change_pixel_pair == 255)]
        unchange_data = source_data[np.where(unchange_pixel_pair == 255)]
        super_idx = cal_max_score(super_center, unchange_data)
        max_super_unchange_center_idx = super_label[
            super_idx]  

        adj_type = superpixel_mark[np.argwhere(adj_matix[
            max_super_unchange_center_idx, :] == 1).squeeze()]  
        

        superpixel_unlabel_data_idx = tuple(
            retain_same_row(
                np.argwhere(segments_map == max_super_unchange_center_idx),
                np.argwhere(traverse_mark == 0)).T)
        if superpixel_unlabel_data_idx:
            superpixel_unlabel_data = source_data[superpixel_unlabel_data_idx]

            d_c = np.mean(np.exp(-cdist(superpixel_unlabel_data, change_data)),
                          axis=1)
            d_uc = np.mean(
                np.exp(-cdist(superpixel_unlabel_data, unchange_data)), axis=1)
            diff_d = np.abs(d_c - d_uc)

            if np.sum(diff_d[d_c < d_uc]) > np.sum(
                    diff_d[d_c >= d_uc]) and np.sum(
                        (d_c < d_uc).astype(int)) > d_c.shape[0] - np.sum(
                            (d_c < d_uc).astype(int)) and (
                                (np.array([1, 1]) not in adj_type)
                                or np.all(adj_type == 0)):
                unchange_pixel_pair[superpixel_unlabel_data_idx] = 255
                superpixel_mark[max_super_unchange_center_idx, :] = np.array(
                    [1, 0])
            else:
                uncertain_pixel_pair[superpixel_unlabel_data_idx] = 255
                superpixel_mark[max_super_unchange_center_idx, :] = np.array(
                    [1, 2])
            traverse_mark[superpixel_unlabel_data_idx] = 1
        else:
            max_super_change_pixel = change_pixel_pair[np.where(
                segments_map == max_super_unchange_center_idx)]
            max_super_unchange_pixel = unchange_pixel_pair[np.where(
                segments_map == max_super_unchange_center_idx)]
            if 0 not in max_super_change_pixel and 255 not in max_super_unchange_pixel:
                superpixel_mark[max_super_unchange_center_idx, :] = np.array(
                    [1, 1])
            elif 0 not in max_super_unchange_pixel and 255 not in max_super_change_pixel:
                superpixel_mark[max_super_unchange_center_idx, :] = np.array(
                    [1, 0])
            else:
                superpixel_mark[max_super_unchange_center_idx, :] = np.array(
                    [1, 2])
        
        super_label = np.delete(super_label, super_idx, axis=0)
        super_center = np.delete(super_center, super_idx, axis=0)

        assign_rate = (
            (unlabel_size - np.argwhere(traverse_mark == 0).shape[0]) * 100 /
            unlabel_size)
        print(
            f'epoch：{epoch}; assign_rate：{assign_rate:.4f} %; time：{(time.time()-init_time):.2f} s'
        )

        if epoch % 1 == 0:
            change_map = np.zeros_like(change_pixel_pair)
            change_map[np.where(change_pixel_pair == 255)] = 255
            change_map[np.where(change_pixel_pair + unchange_pixel_pair +
                                uncertain_pixel_pair == 0)] = 80
            change_map[np.where(uncertain_pixel_pair == 255)] = 160
            imsave(
                os.path.join(args.root_path, args.dataset,
                             f'exp_result-{Ns}-{assign_th}',
                             'iterative_process_image',
                             f'label_map_{epoch}_{assign_rate:.4f}.bmp'),
                change_map.astype('uint8'))
        epoch += 1

    change_map = np.zeros_like(change_pixel_pair)
    change_map[np.where(change_pixel_pair == 255)] = 255
    change_map[np.where(change_pixel_pair + unchange_pixel_pair +
                        uncertain_pixel_pair == 0)] = 80
    change_map[np.where(uncertain_pixel_pair == 255)] = 160
    return change_pixel_pair, unchange_pixel_pair, uncertain_pixel_pair


def denoising(position, win_size=20, th=0.3):
    n = int((win_size - 1) / 2)
    condition_p = position.copy()
    for (i, j) in tqdm(np.argwhere(position == 255)):
        a = max(i - n, 0)
        b = min(i - n + win_size, h)
        c = max(j - n, 0)
        d = min(j - n + win_size, w)
        ratio = np.argwhere(condition_p[a:b, c:d] == 255).shape[0] / ((b - a) *
                                                                      (d - c))
        if ratio < th:
            position[i, j] = 0

    return position


def SLIC(img, n_seg=300, compact=45):
    #compact=35
    segments = slic(img.astype('uint8'),
                    n_segments=n_seg,
                    compactness=compact,
                    enforce_connectivity=True,
                    convert2lab=False)

    num = segments.max() + 1
    superpixel_center = np.zeros((num, img.shape[2]))
    for i in range(num):
        superpixel_center[i, :] = np.mean(img[np.where(segments == i)], axis=0)
    print(f'num of superpixel:{num}  ')
    return segments, superpixel_center


from sklearn.metrics.pairwise import cosine_similarity


def fusion_diff_image(base, co, gen):
    diff1 = np.max(np.abs(base.astype('int') - co.astype('int')), axis=2)
    diff2 = np.max(np.abs(base.astype('int') - gen.astype('int')), axis=2)
    return (diff1 + diff2) / 2


def get_diff_image(base_data, co_data, gen_image):
    diff_image_r = np.abs(base_data[:, :, 0].astype('int') -
                          gen_image[:, :, 0].astype('int'))
    diff_image_g = np.abs(base_data[:, :, 1].astype('int') -
                          gen_image[:, :, 1].astype('int'))
    diff_image_b = np.abs(base_data[:, :, 2].astype('int') -
                          gen_image[:, :, 2].astype('int'))
    
    diff_image = np.max(
        np.abs(base_data.astype('int') - gen_image.astype('int')), axis=2)
    

    if args.dataset == 'Gloucester':
        
        diff_image = (diff_image_b + diff_image_g) / 2
    elif args.dataset == 'Gloucester_':
        diff_image = diff_image_b
        diff_image = (diff_image_b + diff_image_r.max() - diff_image_r) / 2
        diff_image -= diff_image.min()
    elif args.dataset == 'California':
        diff_image = np.abs(base_data[:, :, 1].astype('int') -
                            co_data[:, :, 1].astype('int'))
        diff_image = (diff_image + diff_image_g) / 2
        diff_image = diff_image_b
        
    elif args.dataset == 'Texas':
        diff_image = fusion_diff_image(base_data, co_data, gen_image)

    imsave(os.path.join(args.root_path, args.dataset, 'diff_image.png'),
           diff_image.astype('uint8'))

    return diff_image


def get_evaluate_result(image1, image2, unchange_label, change_label,
                        uncertain_label, true_mask):
    result_rf, result_knn, result_svm = classify_get_change_map(
        image1, image2, unchange_label, change_label)
    print('rf:')
    result = result_rf
    result = median(result, 3)
    evaluate_result = CDTool.evaluate(result, true_mask)
    result[result == 1] = 255
    imsave(
        os.path.join(args.root_path, args.dataset,
                     f'exp_result-{Ns}-{assign_th}',
                     f'change_map_{evaluate_result[-1]}_rf.bmp'),
        result.astype('uint8'))
    # save the result to .txt
    file = open(
        os.path.join(args.root_path, args.dataset,
                     f'exp_result-{Ns}-{assign_th}', f'result.txt'), 'w')
    for s in evaluate_result:
        file.write(str(s) + '\t')
    file.write('\n')

    print('knn:')
    result = result_knn
    result = median(result, 3)
    evaluate_result = CDTool.evaluate(result, true_mask)
    result[result == 1] = 255
    imsave(
        os.path.join(args.root_path, args.dataset,
                     f'exp_result-{Ns}-{assign_th}',
                     f'change_map_{evaluate_result[-1]}_knn.bmp'),
        result.astype('uint8'))
    for s in evaluate_result:
        file.write(str(s) + '\t')
    file.write('\n')

    print('svm:')
    result = result_svm
    result = median(result, 3)
    evaluate_result = CDTool.evaluate(result, true_mask)
    result[result == 1] = 255
    imsave(
        os.path.join(args.root_path, args.dataset,
                     f'exp_result-{Ns}-{assign_th}',
                     f'change_map_{evaluate_result[-1]}_svm.bmp'),
        result.astype('uint8'))
    for s in evaluate_result:
        file.write(str(s) + '\t')
    file.write('\n')


    result_rf, result_knn, result_svm = classify_get_change_map_3(
        image1, image2, unchange_label, change_label, uncertain_label)
    print('rf:')
    result = result_rf
    result = median(result, 3)
    e_evaluate_result = CDTool.evidence_evaluate(result, true_mask)
    result[result == 1] = 255
    result[result == 2] = 128
    imsave(
        os.path.join(args.root_path, args.dataset,
                     f'exp_result-{Ns}-{assign_th}',
                     f'change_map_meta_{e_evaluate_result[-1]}_rf.bmp'),
        result.astype('uint8'))
    for s in e_evaluate_result:
        file.write(str(s) + '\t')
    file.write('\n')

    print('knn:')
    result = result_knn
    result = median(result, 3)
    e_evaluate_result = CDTool.evidence_evaluate(result, true_mask)
    result[result == 1] = 255
    result[result == 2] = 128
    imsave(
        os.path.join(args.root_path, args.dataset,
                     f'exp_result-{Ns}-{assign_th}',
                     f'change_map_meta_{e_evaluate_result[-1]}_knn.bmp'),
        result.astype('uint8'))
    for s in e_evaluate_result:
        file.write(str(s) + '\t')
    file.write('\n')

    print('svm:')
    result = result_svm
    result = median(result, 3)
    e_evaluate_result = CDTool.evidence_evaluate(result, true_mask)
    result[result == 1] = 255
    result[result == 2] = 128
    imsave(
        os.path.join(args.root_path, args.dataset,
                     f'exp_result-{Ns}-{assign_th}',
                     f'change_map_meta_{e_evaluate_result[-1]}_svm.bmp'),
        result.astype('uint8'))
    for s in e_evaluate_result:
        file.write(str(s) + '\t')
    file.write('\n')
    file.close()


def main(time1, time2, true, gen_img):
    print('Begin load data...')
    image1 = cv2.imread(time1)
    image2 = cv2.imread(time2)
    image1_gray = cv2.imread(time1, 0)
    image2_gray = cv2.imread(time2, 0)
    true_mask = cv2.imread(true, 0)
    global h, w, b
    h, w, b = image1.shape

    gen_image_gray = cv2.imread(gen_img, 0)
    gen_image = cv2.imread(gen_img)
    if args.dataset == 'River' or args.dataset == 'Gloucester_' or args.dataset == 'California':
        # pre-event:optical   post-event:SAR
        base_data = image1.copy()
        co_data = image2.copy()
    else:
        # pre-event:SAR   post-event:optical
        co_data = image1.copy()
        base_data = image2.copy()

    # load parameters
    parameters_dict = get_config()
    diff_threshold_c = parameters_dict[args.dataset][0]
    diff_threshold_uc = parameters_dict[args.dataset][1]
    win_size_c = parameters_dict[args.dataset][2]
    th_c = parameters_dict[args.dataset][3]
    win_size_uc = parameters_dict[args.dataset][4]
    th_uc = parameters_dict[args.dataset][5]
    global Ns, assign_th
    Ns = parameters_dict[args.dataset][6]
    assign_th = parameters_dict[args.dataset][7]  
    make_dir(
        os.path.join(args.root_path, args.dataset,
                     f'exp_result-{Ns}-{assign_th}',
                     'iterative_process_image'))
    file = open(
        os.path.join(args.root_path, args.dataset,
                     f'exp_result-{Ns}-{assign_th}', f'parameters.txt'), 'w')
    for s in parameters_dict[args.dataset]:
        file.write(str(s) + '\t')
    file.close()

    # 1.CAE+Adain  getting different images after transferred 
    diff_image = get_diff_image(base_data, co_data, gen_image)

    # 2.getting the reliable changed/unchanged pairs
    print('Begin find significant samples...')

    c_position = diff_image.copy()
    c_position[c_position < diff_threshold_c] = 0
    c_position[c_position != 0] = 255
    imsave(os.path.join(args.root_path, args.dataset, 'c_position.png'),
           c_position.astype('uint8'))
    change_pixel = denoising(c_position, win_size_c, th_c)
    print(f'change:{np.argwhere(change_pixel == 255).shape[0]}')
    imsave(os.path.join(args.root_path, args.dataset, 'change_pixel.png'),
           change_pixel.astype('uint8'))

    uc_position = diff_image.copy()
    uc_position[uc_position > diff_threshold_uc] = 200
    uc_position[uc_position != 200] = 255
    uc_position[uc_position == 200] = 0
    imsave(os.path.join(args.root_path, args.dataset, 'uc_position.png'),
           uc_position.astype('uint8'))
    unchange_pixel = denoising(uc_position, win_size_uc, th_uc)
    #unchange_pixel = median(unchange_pixel.astype('uint8'), 3)
    print(f'unchange:{np.argwhere(unchange_pixel == 255).shape[0]}')
    imsave(os.path.join(args.root_path, args.dataset, 'unchange_pixel.png'),
           unchange_pixel.astype('uint8'))
    
    # 3. iteration superpixel and assign labels
    print('Begin SLIC...')
    source_data = np.concatenate((base_data, gen_image), axis=2)
    segments, super_center = SLIC(source_data,
                                  n_seg=int(h * w / Ns),
                                  compact=0.25)
    adj_mat = np.array(adj_matrix(
        graph.RAG(segments)).todense())  
    
    print('Begin iterative assign label...')
    change_label, unchange_label, uncertain_label = iterative_superpixel(
        source_data.astype('int'), change_pixel.astype('int'),
        unchange_pixel.astype('int'), segments, super_center, adj_mat,
        assign_th)

    imsave(
        os.path.join(args.root_path, args.dataset,
                     f'exp_result-{Ns}-{assign_th}', 'change_label.png'),
        change_label.astype('uint8'))
    imsave(
        os.path.join(args.root_path, args.dataset,
                     f'exp_result-{Ns}-{assign_th}', 'unchange_label.png'),
        unchange_label.astype('uint8'))

    change_label[change_label == 255] = 1
    unchange_label[unchange_label == 255] = 1
    uncertain_label[uncertain_label == 255] = 1
    ##################################  classify   ##################################
    print('8.classify.....')
    print(f'train data: change-({change_label.sum()})',
          f'unchange-({unchange_label.sum()})',
          f'uncertain-({uncertain_label.sum()})')
    true_mask[true_mask < 120] = 0
    true_mask[true_mask != 0] = 1

    change_map = get_change_map(image1,
                                image2,
                                unchange_label,
                                change_label,
                                case='svm')
    result = change_map
    result = median(result, 3)
    evaluate_result = CDTool.evaluate(result, true_mask)
    result[result == 1] = 255
    imsave(
        os.path.join(args.root_path, args.dataset,
                     f'exp_result-{Ns}-{assign_th}',
                     f'change_map_{evaluate_result[-1]}.bmp'),
        result.astype('uint8'))
    
    file = open(
        os.path.join(args.root_path, args.dataset,
                     f'exp_result-{Ns}-{assign_th}', f'result.txt'), 'w')
    for s in evaluate_result:
        file.write(str(s) + '\t')
    file.write('\n')

    change_map = get_change_map_3(image1,
                                  image2,
                                  unchange_label,
                                  change_label,
                                  uncertain_label,
                                  case='svm')
    result = change_map
    result = median(result, 3)
    e_evaluate_result = CDTool.evidence_evaluate(result, true_mask)
    result[result == 1] = 255
    result[result == 2] = 128
    imsave(
        os.path.join(args.root_path, args.dataset,
                     f'exp_result-{Ns}-{assign_th}',
                     f'change_map_meta_{e_evaluate_result[-1]}.bmp'),
        result.astype('uint8'))
    for s in e_evaluate_result:
        file.write(str(s) + '\t')
    file.write('\n')
    file.close()

    print('ok')


if __name__ == "__main__":
    print('root_path:', args.root_path, 'dataset:', args.dataset)

    cur_dataset_path = os.path.join(args.dataset_path, args.dataset)
    data_file = [
        f for f in os.listdir(cur_dataset_path)
        if os.path.isfile(os.path.join(cur_dataset_path, f))
    ]

    t1 = os.path.join(cur_dataset_path, data_file[0])
    t2 = os.path.join(cur_dataset_path, data_file[1])
    truth = os.path.join(cur_dataset_path, data_file[2])
    gen_image = os.path.join(args.root_path, 'cae_result', args.dataset,
                             'gen_image.png')
    main(t1, t2, truth, gen_image)
