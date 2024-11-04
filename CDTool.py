import numpy as np
from imageio import imread, imsave
from tqdm import tqdm
from sklearn import metrics
import os


def norm_0_1(data):
    if len(data.shape) == 3:
        data = data[:, :, 0]
    if data.max() == 255:
        data[data < 120] = 0
        data[data != 0] = 1

    return data


def evaluate(change_map, true_mask):
    change_map = norm_0_1(change_map)
    true_mask = norm_0_1(true_mask)
    # 0  1
    matrix = metrics.confusion_matrix(true_mask.reshape(-1, 1),
                                      change_map.reshape(-1, 1))
    kappa = metrics.cohen_kappa_score(true_mask.reshape(-1, 1),
                                      change_map.reshape(-1, 1))
    TP = matrix[1, 1]
    TN = matrix[0, 0]
    FP = matrix[0, 1]
    FN = matrix[1, 0]
    Re = (FP + FN) / (TP + TN + FP + FN)
    Ra = (TP + TN) / (TP + TN + FP + FN)
    Rp = TP / (TP + FP)
    Rm = FN / (TP + FN)
    Rf = FP / (TP + FP)

    Nc = FN + TP
    Nu = FP + TN
    PRE = ((TP + FP) * Nc + (FN + TN) * Nu) / ((TP + TN + FP + FN) *
                                               (TP + TN + FP + FN))
    Kappa1 = (Ra - PRE) / (1 - PRE)
    kappa = kappa

    print(f'TN: {TN}', f'TP: {TP}', f'FN: {FN}', f'FP: {FP}', f'Re: {Re:.4f}',
          f'Ra: {Ra:.4f}', f'Rp: {Rp:.4f}', f'Rm: {Rm:.4f}', f'Rf: {Rf:.4f}',
          f'kappa: {kappa:.4f}')

    Re = float(f'{Re:.4f}')
    Ra = float(f'{Ra:.4f}')
    Rp = float(f'{Rp:.4f}')
    Rm = float(f'{Rm:.4f}')
    Rf = float(f'{Rf:.4f}')
    kappa = float(f'{kappa:.4f}')
    return [TN, TP, FN, FP, Re, Ra, Rp, Rm, Rf, kappa]


def evidence_evaluate(change_map, true_mask, gamma=0.8):
    # 0  1  2
    A = 2
    matrix = metrics.confusion_matrix(true_mask.reshape(-1, 1),
                                      change_map.reshape(-1, 1))

    ETP = matrix[1, 1] + matrix[1, 2].sum() * (1 / A)**gamma
    ETN = matrix[0, 0] + matrix[0, 2].sum() * (1 / A)**gamma
    FP = matrix[0, 1]
    FN = matrix[1, 0]
    Ra = (ETP + ETN) / (ETP + ETN + FP + FN)
    Re = (FP + FN) / matrix.sum()
    Ri = matrix[:, 2].sum() / matrix.sum()

    Nc = FN + ETP
    Nu = FP + ETN
    PRE = ((ETP + FP) * Nc + (FN + ETN) * Nu) / ((ETP + ETN + FP + FN) *
                                                 (ETP + ETN + FP + FN))
    Kappa = (Ra - PRE) / (1 - PRE)
    print(f'ETN: {ETN:.2f}', f'ETP: {ETP:.2f}', f'EFN: {FN}', f'EFP: {FP}',
          f'Re: {Re:.4f}', f'Ri: {Ri:.4f}', f'ERa: {Ra:.4f}',
          f'E_kappa: {Kappa:.4f}')

    ETN = float(f'{ETN:.2f}')
    ETP = float(f'{ETP:.2f}')
    Re = float(f'{Re:.4f}')
    Ra = float(f'{Ra:.4f}')
    Ri = float(f'{Ri:.4f}')
    Kappa = float(f'{Kappa:.4f}')
    return [ETN, ETP, FN, FP, Re, Ri, Ra, Kappa]


def display(changemap, true_mask):
    # 0  1
    Final_changemap = np.zeros((true_mask.shape[0], true_mask.shape[1], 3))
    for i in range(true_mask.shape[0]):
        for j in range(true_mask.shape[1]):
            if changemap[i, j] == 1 and true_mask[i, j] == 1:
                Final_changemap[i, j, 0] = 255
                Final_changemap[i, j, 1] = 255
                Final_changemap[i, j, 2] = 255
            elif changemap[i, j] == 0 and true_mask[i, j] == 1:
                Final_changemap[i, j, 0] = 255
            elif changemap[i, j] == 1 and true_mask[i, j] == 0:
                Final_changemap[i, j, 1] = 255
    return Final_changemap


def test(TN, TP, FN, FP):
    # 计算 ra，kappa
    pcc = (TP + TN) / (TP + TN + FP + FN)

    Nc = FN + TP
    Nu = FP + TN
    PRE = ((TP + FP) * Nc + (FN + TN) * Nu) / ((TP + TN + FP + FN) *
                                               (TP + TN + FP + FN))
    Ka = (pcc - PRE) / (1 - PRE)

    return round(pcc, 4), round(Ka, 4)


def sign_rg(changemap, truth, savepath, ka):
    if len(changemap.shape) == 3:
        changemap = changemap[:, :, 0]
    if len(truth.shape) == 3:
        truth = truth[:, :, 0]

    Final = np.zeros((truth.shape[0], truth.shape[1], 3))
    truth[truth > 120] = 255
    truth[truth != 255] = 0
    x=changemap.max()
    changemap[changemap > 120] = 255
    changemap[changemap != 255] = 0
    x=changemap.max()
    for i in tqdm(range(truth.shape[0])):
        for j in range(truth.shape[1]):
            if changemap[i, j] == 255 and truth[i, j] == 255:
                Final[i, j, 0] = 255
                Final[i, j, 1] = 255
                Final[i, j, 2] = 255
            elif changemap[i, j] == 0 and truth[i, j] == 255:
                Final[i, j, 0] = 255
            elif changemap[i, j] == 255 and truth[i, j] == 0:
                Final[i, j, 1] = 255
    imsave(os.path.join(savepath, f'change_map_sccn_{ka}_rg.bmp'),
           Final.astype('uint8'))

    # im = Image.fromarray(np.uint8(Final))
    # im.save(os.path.join(savepath, f'change_map_{method}_data{datanumber}_rg.pdf'))
    print('ok')


if __name__ == '__main__':

    cm = imread(r'SCCN\sccn\change_map_sccn_0.6923.bmp')
    th = imread(r'used_data\Gloucester\truth.png')

    result = evaluate(cm.copy(), th.copy())

    # pcc, Ka = test(TP=5495, TN=91578, FP=1997, FN=931)
    # print(pcc, Ka)

    s_path = r'SCCN\sccn'
    sign_rg(cm.copy(), th.copy(), s_path, ka=result[-1])
