from datasets.dataloader import LIVEDatasetLoader, TID2013DatasetLoader, TID2013_GTtable
import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.misc as m

from torch.utils.data import DataLoader
from torch.autograd import Variable

from models.deepQA import deepIQA_model as predictNet

import skimage.measure


def showfigures_TID13():
    def __find_test_img(gt_score):
        gt_table, _, _ = TID2013_GTtable('../data/TID2013_dataset/mos_with_names.txt')

        for idx, sample in enumerate(gt_table):

            if gt_score == float(sample[1]):
                img_path = '../data/TID2013_dataset/distorted_images/%s' % (gt_table[idx, 0])
                ref_path = '../data/TID2013_dataset/reference_images/I%s.BMP' % (sample[0][1:3])

        img = m.imread(img_path)
        ref = m.imread(ref_path)

        return img, ref

    testset = TID2013DatasetLoader('../data/TID2013_dataset/',
                                   train_phase=False,
                                   is_sample=False,
                                   seed=12,
                                   global_permute=False)
    testloader = DataLoader(testset,
                            shuffle=False,
                            batch_size=1,
                            num_workers=1,
                            pin_memory=True)

    # load the trained model
    model = predictNet()
    model_dict = torch.load('snapshots/deepQA_TID13_seed32_0.9286_0.9244_epoch3780.pth')['model']
    model.load_state_dict(model_dict)  # copyWeights(model, model_dict, freeze=False)
    model.eval()
    model.to('cuda')

    for batch_id, (img, error, score_gt) in enumerate(testloader):
        score_gt = score_gt.type('torch.FloatTensor')
        img, error, score_gt, = Variable(img.cuda()), \
                                Variable(error.cuda()), \
                                Variable(score_gt.cuda())

        score_pred, senMap = model(img, error)
        # print(score_pred.data.cpu().numpy(), score_gt.data.cpu().numpy())
        # print(senMap.shape)

        score_pred_np = score_pred.data.cpu().numpy()
        score_gt_np = score_gt.data.cpu().numpy()

        print(batch_id, score_pred_np, score_gt_np)
        img_ori, ref_ori = __find_test_img(score_gt_np)

        img_np = img.data.cpu().numpy()
        error_np = error.data.cpu().numpy()
        senMap_np = senMap.data.cpu().numpy()

        error_np = np.squeeze(error_np)
        img_np = np.squeeze(img_np)

        error_np_resize = skimage.measure.block_reduce(error_np, (4, 4), np.mean)

        perceptual = error_np_resize*senMap_np

        plt.figure(figsize=(12, 8))
        plt.suptitle('GT:%.4f, Predit:%.4f' % (score_gt_np, score_pred_np),
                     fontsize=16)

        plt.subplot(231)
        plt.imshow(img_ori)
        plt.xlabel('Distorted Img', fontsize=14)
        plt.xticks([])
        plt.yticks([])

        plt.subplot(234)
        plt.imshow(ref_ori)
        plt.xlabel('Reference Img', fontsize=14)
        plt.xticks([])
        plt.yticks([])

        plt.subplot(232)
        plt.imshow(img_np)
        plt.xlabel('Per-processed Img', fontsize=14)
        plt.xticks([])
        plt.yticks([])

        plt.subplot(233)
        plt.imshow(error_np_resize)
        plt.xlabel('Error Map', fontsize=14)
        plt.xticks([])
        plt.yticks([])

        plt.subplot(235)
        plt.imshow(senMap_np)
        plt.xlabel('Sensitivity Map', fontsize=14)
        plt.xticks([])
        plt.yticks([])

        plt.subplot(236)
        plt.imshow(perceptual)
        plt.xlabel('Perceptual Map', fontsize=14)
        plt.xticks([])
        plt.yticks([])

        plt.tight_layout()

        if batch_id == 4:
            plt.savefig('TID13_exp2.png', dpi=500)

        plt.show()

        if batch_id > 25:
            break


def showfigures_LIVE():
    testset = LIVEDatasetLoader('../data/LIVE_dataset/',
                                train_phase=False,
                                is_sample=False,
                                seed=12,
                                global_permute=False)
    testloader = DataLoader(testset,
                            shuffle=False,
                            batch_size=1,
                            num_workers=1,
                            pin_memory=True)

    # load the trained model
    model = predictNet()
    model_dict = torch.load('snapshots/deepQA_LIVE_seed12_0.9708_0.9665_epoch2430.pth')['model']
    model.load_state_dict(model_dict)  # copyWeights(model, model_dict, freeze=False)
    model.eval()
    model.to('cuda')

    for batch_id, (img, error, score_gt) in enumerate(testloader):
        score_gt = score_gt.type('torch.FloatTensor')
        img, error, score_gt, = Variable(img.cuda()), \
                                Variable(error.cuda()), \
                                Variable(score_gt.cuda())

        score_pred, senMap = model(img, error)
        # print(score_pred.data.cpu().numpy(), score_gt.data.cpu().numpy())
        # print(senMap.shape)

        score_pred_np = score_pred.data.cpu().numpy()
        score_gt_np = score_gt.data.cpu().numpy()

        print(batch_id, score_pred_np, score_gt_np)

        img_np = img.data.cpu().numpy()
        error_np = error.data.cpu().numpy()
        senMap_np = senMap.data.cpu().numpy()

        error_np = np.squeeze(error_np)
        img_np = np.squeeze(img_np)

        error_np_resize = skimage.measure.block_reduce(error_np, (4, 4), np.mean)

        perceptual = error_np_resize*senMap_np

        plt.figure(figsize=(12, 8))
        plt.suptitle('GT:%.4f, Predit:%.4f' % (score_gt_np*10, score_pred_np*10),
                     fontsize=16)

        plt.subplot(231)
        plt.imshow(m.imread('../data/LIVE_dataset/studentsculpture/3img108-77.1967-GB.jpg'))
        plt.xlabel('Distorted Img', fontsize=14)
        plt.xticks([])
        plt.yticks([])

        plt.subplot(234)
        plt.imshow(m.imread('../data/LIVE_dataset/studentsculpture/1studentsculptureOriginal.jpg'))
        plt.xlabel('Reference Img', fontsize=14)
        plt.xticks([])
        plt.yticks([])

        plt.subplot(232)
        plt.imshow(img_np)
        plt.xlabel('Per-processed Img', fontsize=14)
        plt.xticks([])
        plt.yticks([])

        plt.subplot(233)
        plt.imshow(error_np_resize)
        plt.xlabel('Error Map', fontsize=14)
        plt.xticks([])
        plt.yticks([])

        plt.subplot(235)
        plt.imshow(senMap_np)
        plt.xlabel('Sensitivity Map', fontsize=14)
        plt.xticks([])
        plt.yticks([])

        plt.subplot(236)
        plt.imshow(perceptual)
        plt.xlabel('Perceptual Map', fontsize=14)
        plt.xticks([])
        plt.yticks([])

        plt.tight_layout()

        if batch_id == 12:
            plt.savefig('LIVE_exp2.png', dpi=500)

        plt.show()

        if batch_id > 15:
            break







if __name__=='__main__':
    # showfigures_LIVE()
    showfigures_TID13()