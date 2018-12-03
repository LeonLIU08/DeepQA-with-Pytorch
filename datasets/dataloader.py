import os
import math
import collections
import torch
import torchvision.transforms as transforms
import numpy as np
import scipy.misc as m
from skimage.color import rgb2gray
from PIL import Image
import random
from torch.utils import data
import torch.nn as nn
from torch.autograd import Variable
from preprocessing import error_map, low_frequency_sub

torch.set_default_tensor_type('torch.FloatTensor')


def gci(filepath, ext='.jpg'):
    def __gci(filepath):
        files = os.listdir(filepath)
        for fi in files:
            fi_d = os.path.join(filepath, fi)
            if os.path.isdir(fi_d):
                __gci(fi_d)
            else:
                if os.path.splitext(fi_d)[1] == '.jpg':
                    file_list.append(fi_d)

    file_list = []
    __gci(filepath)
    return file_list


def TID2013_GTtable(gt_file):
    table = []
    score_all = []
    with open(gt_file) as f:
        lines = f.readlines()

    for line in lines:
        score = float(line.split(' ')[0])
        img_name = line.split(' ')[1][:-2]
        table.append([img_name, score])
        score_all.append(score)

    score_all = np.array(score_all)
    score_min = np.min(score_all)
    score_max = np.max(score_all-score_min)

    return np.array(table), score_min, score_max


class TID2013DatasetLoader(data.Dataset):
    def __init__(self, root='../../data/TID2013_dataset/',
                 transform_prob=0.,
                 img_size=(384, 384),
                 train_phase=True,
                 is_sample=True,
                 seed=42,
                 global_permute=True):
        gt_table, self.score_min, self.score_max = TID2013_GTtable(root + 'mos_with_names.txt')
        self.root = root
        self.is_sample = is_sample
        self.img_size = img_size
        self.transform_prob = transform_prob
        self.files = collections.defaultdict(list)
        # self.trans = transforms.Compose([
        #     transforms.RandomHorizontalFlip(),
        #     # transforms.RandomVerticalFlip(),
        #     transforms.ToTensor(),
        # ])

        trainnum = int(len(gt_table) * 0.8)
        if global_permute:
            gt_table = np.random.RandomState(seed=seed).permutation(gt_table)
            tra_l = gt_table[0:trainnum]
            tes_l = gt_table[trainnum:]
        else:
            tra_l, tes_l = self.ref_permutation(gt_table, seed)

        if train_phase:
            self.phase = 'train'
            tra_l = np.random.RandomState(seed=seed).permutation(tra_l)
            self.files[self.phase] = tra_l
        else:
            self.phase = 'test'
            tes_l = np.random.RandomState(seed=seed).permutation(tes_l)
            self.files[self.phase] = tes_l

        n_images = len(self.files[self.phase])
        print("{} number: {:d}".format(self.phase, n_images))

    def ref_permutation(self, gt_table, seed=42):
        ref_idx = np.arange(1, 25)
        ref_idx = np.random.RandomState(seed=seed).permutation(ref_idx)

        tra_idx = ref_idx[:20]
        # tes_idx = ref_idx[20:]

        tra_table = []
        tes_table = []
        for i in gt_table:
            img_ref = i[0][1:3]
            if int(img_ref) in tra_idx:
                tra_table.append(i)
            else:
                tes_table.append(i)

        tra_table = np.random.RandomState(seed=seed).permutation(np.array(tra_table))
        tes_table = np.random.RandomState(seed=seed).permutation(np.array(tes_table))

        return tra_table, tes_table

    def __len__(self):
        return len(self.files[self.phase])

    def __getitem__(self, index):
        img_info = self.files[self.phase][index]

        # Load image and ref
        img = m.imread(self.root+'distorted_images/'+img_info[0])
        ref_idx = img_info[0][1:3]
        img0 = m.imread(self.root+'reference_images/I%s.BMP' % ref_idx)

        if self.is_sample:
            rmax, cmax = img.shape[0]-self.img_size[0], img.shape[1]-self.img_size[1]
            rb, cb = random.randint(0, rmax), random.randint(0, cmax)
            img = img[rb:rb+self.img_size[0], cb:cb+self.img_size[1], :]
            img0 = img0[rb:rb + self.img_size[0], cb:cb + self.img_size[1], :]
        else:
            tmpimgsize = (int(img.shape[0]/16.)*16, int(img.shape[1]/16.)*16)
            img = m.imresize(img, tmpimgsize, interp='nearest')
            img0 = m.imresize(img0, tmpimgsize, interp='nearest')

        img_gray = rgb2gray(img)
        img0_gray = rgb2gray(img0)
        img_d = low_frequency_sub(img_gray*255)
        img_r = low_frequency_sub(img0_gray*255)

        error = error_map(img_d, img_r, epsilon=1.)

        # Generate score
        tscore = float(img_info[1])
        # tscore = (tscore - self.score_min) / self.score_max * 100
        score = tscore

        if self.phase == 'train':
            if np.random.rand() < self.transform_prob:
                # horizontally flipping only
                img_d = np.flip(img_d, axis=1).copy()
                error = np.flip(error, axis=1).copy()

        # error = m.imresize(error, (img.shape[0]/4, img.shape[1]/4),
        #                    interp='nearest')/255.
        # print('error range', error.max(), error.min())
        img_d = Image.fromarray(img_d)
        img_d = transforms.ToTensor()(img_d)
        error = torch.from_numpy(error).float()

        return img_d, error, score


class LIVEDatasetLoader(data.Dataset):
    def __init__(self, root='../../data/LIVE_dataset/',
                 transform_prob=0.5,
                 img_size=(384, 384),
                 train_phase=True,
                 is_sample=True,
                 seed=42,
                 global_permute=True):
        self.root = root
        self.is_sample = is_sample
        self.img_size = img_size
        # self.score_size = (int(img_size[0]//4), int(img_size[1]//4))
        self.transform_prob = transform_prob
        self.files = collections.defaultdict(list)
        # self.trans = transforms.Compose([
        #     transforms.RandomHorizontalFlip(),
        #     transforms.RandomVerticalFlip(),
        #     transforms.ToTensor(),
        # ])
        # self.namelist = ['bikes','cemetry','house','ocean','sail1','stream',
        #                  'buildings','churchandcapitol','lighthouse','paintedhouse','sail2','studentsculpture',
        #                  'buildings2','coinsinfountain','lighthouse2','parrots','sail3','woman',
        #                  'caps','dancers','manfishing','plane','sail4','womanhat',
        #                  'carnivaldolls','flowersonih35','monarch','rapids','statue']
        img_list = gci(root)
        trainnum = int(len(img_list) * 0.8)

        if global_permute:
            img_list = np.random.RandomState(seed=seed).permutation(img_list)
            tra_l = img_list[0:trainnum]
            tes_l = img_list[trainnum:]
        else:
            tra_l, tes_l = self.ref_permutation(img_list, seed)

        if train_phase:
            self.phase = 'train'
            tra_l = np.random.RandomState(seed=seed).permutation(tra_l)
            self.files[self.phase] = tra_l
        else:
            self.phase = 'test'
            tes_l = np.random.RandomState(seed=seed).permutation(tes_l)
            self.files[self.phase] = tes_l

        n_images = len(self.files[self.phase])
        print("{} number: {:d}".format(self.phase, n_images))

    def ref_permutation(self, image_list, seed):
        # print image_list
        ref_list = []
        for image in image_list:
            ref = image.split('/')[-2]
            if ref not in ref_list:
                ref_list.append(ref)

        ref_len = len(ref_list)

        ref_list = np.random.RandomState(seed=seed).permutation(ref_list)

        # tes_ref = ref_list[int(ref_len*0.8):]
        tra_ref = ref_list[:int(ref_len*0.8)]

        tra_list = []
        tes_list = []
        for image in image_list:

            ref = image.split('/')[-2]
            if ref in tra_ref:
                tra_list.append(image)
            else:
                tes_list.append(image)

        return tra_list, tes_list

    def __len__(self):
        return len(self.files[self.phase])

    def __getitem__(self, index):
        img_path = self.files[self.phase][index]

        # Load image
        img = m.imread(img_path)
        ref_name = img_path.split('/')[-2]
        ref_path = '%s/%s/1%sOriginal.jpg' % (self.root, ref_name, ref_name)
        img0 = m.imread(ref_path)

        if self.is_sample:
            rmax, cmax = img.shape[0]-self.img_size[0], img.shape[1]-self.img_size[1]
            rb, cb = random.randint(0, rmax), random.randint(0, cmax)
            img = img[rb:rb+self.img_size[0], cb:cb+self.img_size[1], :]
            img0 = img0[rb:rb + self.img_size[0], cb:cb + self.img_size[1], :]
        else:
            tmpimgsize = (int(img.shape[0]/16.)*16, int(img.shape[1]/16.)*16)
            img = m.imresize(img, tmpimgsize, interp='nearest')
            img0 = m.imresize(img0, tmpimgsize, interp='nearest')

        img_gray = rgb2gray(img)
        img0_gray = rgb2gray(img0)
        img_d = low_frequency_sub(img_gray * 255)
        img_r = low_frequency_sub(img0_gray * 255)

        error = error_map(img_d, img_r, epsilon=1.)

        # Generate score
        (_, tempfilename) = os.path.split(img_path)
        tempfilename = tempfilename.split('-')
        if len(tempfilename) == 3:
            score = float(tempfilename[1])
        else:
            score = 85.

        if self.phase == 'train':
            if np.random.rand() < self.transform_prob:
                # horizontally flipping only
                img_d = np.flip(img_d, axis=1).copy()
                error = np.flip(error, axis=1).copy()

        # error = m.imresize(error, (img.shape[0]/4, img.shape[1]/4),
        #                    interp='nearest')/255.
        # print('error range', error.max(), error.min())
        img_d = Image.fromarray(img_d)
        img_d = transforms.ToTensor()(img_d)
        error = torch.from_numpy(error).float()

        return img_d, error, score/10.


if __name__=='__main__':
    import matplotlib.pyplot as plt
    import visdom
    import numpy as np

    vis = visdom.Visdom(server='http://localhost', port=8097, env='display')
    img_rows, img_cols = 384, 384
    win_image = vis.image(np.ndarray((3, img_rows, img_cols)), opts=dict(title='Image'))
    win_error = vis.heatmap(np.ndarray((img_rows, img_cols)), opts=dict(title='Score'))

    local_path = '../../data/LIVE_dataset/'
    dst = LIVEDatasetLoader(local_path,
                               img_size=(img_rows, img_cols),
                               train_phase=False,
                               transform_prob=0.5,
                               is_sample=False,
                               global_permute=False)
    trainloader = data.DataLoader(dst, batch_size=1, shuffle=True, num_workers=1, pin_memory=True)

    for i, data in enumerate(trainloader):
        img, error, score = data
        img, score = img[0], score[0].numpy().astype('float32')
        print(img.shape)
        print(error.shape)
        # print(score.mean())
        print(score)

        # Show results
        print(type(img.numpy()))
        print(img.numpy().shape)
        print('img range', img.numpy().max(), img.numpy().min())
        print('err range', error.numpy().max(), error.numpy().min())
        vis.image(img.numpy(), opts=dict(title='Image'), win=win_image)
        vis.heatmap(error.numpy()[0], opts=dict(title='Score'), win=win_error)

        wait = raw_input('PRESS KEY')
        print('')
