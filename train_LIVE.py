from __future__ import unicode_literals
import random
import numpy as np
import os
import argparse
import math
import visdom
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
# from torchvision.utils import make_grid

from models.deepQA import deepIQA_model as predictNet
# from utils.lr_scheduling import poly_lr_scheduler
from utils.validate import val
from datasets.dataloader import LIVEDatasetLoader

home_dir = os.path.dirname(os.path.realpath(__file__))
torch.set_default_tensor_type('torch.FloatTensor')


def parse_args():
    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("prefix",
                        help="Prefix to identify current experiment")

    parser.add_argument("--LIVE_dataset_dir", default='../data/LIVE_dataset/',
                        help="A directory containing img (Images) and cls (GT Segmentation) folder")

    parser.add_argument("--TID2013_dataset_dir", default='../data/TID2013_dataset/',
                        help="A directory containing img (Images) and cls (GT Segmentation) folder")

    parser.add_argument("--max_epoch", default=3000, type=int,
                        help="Maximum iterations.")

    parser.add_argument("--snapshot_dir", default=os.path.join(home_dir, 'snapshots'),
                        help="Location to store the snapshot")

    parser.add_argument("--batch_size", default=48, type=int,
                        help="Batch size for training")

    parser.add_argument("--lr", default=5e-4, type=float,
                        help="lr for discriminator")

    parser.add_argument("--seed", default=42, type=int,
                        help="random seed for permutation")

    parser.add_argument("--step", default=1, type=int,
                        help="Training step")

    parser.add_argument("--env", default='LIVE',
                        help="Name of visdom environment")

    args = parser.parse_args()

    return args


class visualize():
    def __init__(self, args, imh=432, imw=432):
        # Setup visdom for visualization

        self.imh = imh
        self.imw = imw
        self.vis = visdom.Visdom(server='http://localhost', port=8097, env=args.env)
        self.losssco_win = self.vis.line(X=np.zeros(1),
                                         Y=np.zeros(1),
                                         opts=dict(xlabel='Epoch',
                                                   ylabel='Loss',
                                                   title='Score Loss'))
        self.correlation_win = self.vis.line(X=np.zeros(1),
                                             Y=np.zeros(1),
                                             opts=dict(xlabel='Epoch',
                                                       ylabel='Cor',
                                                       title='Correlation'))
        self.scatter_win = self.vis.scatter(X=np.random.rand(100, 2),
                                            Y=np.zeros(100)+1,
                                            opts=dict(xtickmin=0,
                                                      xtickmax=10,
                                                      ytickmin=0,
                                                      ytickmax=10,
                                                      xtickstep=1,
                                                      ytickstep=1,
                                                      markersymbol='dot',
                                                      markersize=4))
        # heatmap will vertically flip the image
        self.sen_win = self.vis.heatmap(np.ones((imh, imw)), opts=dict(title='Sen Map'))
        self.img_win = self.vis.heatmap(np.ones((imh, imw)), opts=dict(title='Image'))
        self.err_win = self.vis.heatmap(np.ones((imh, imw)), opts=dict(title='Error'))

    def update(self,
               losssco=None,
               lcc=None,
               srocc=None,
               test_loss=None,
               senMap=None,
               img=None,
               errMap=None,
               epoch=None,
               result_list=None):

        # losssco = np.array(losssco).mean()
        result_list = np.transpose(result_list)
        curr_epoch = max(epoch)
        # print(result_list.shape)
        self.vis.line(X=epoch,
                      Y=np.array(losssco),
                      win=self.losssco_win,
                      name='tra',
                      update='new')
        self.vis.line(X=epoch,
                      Y=np.array(test_loss),
                      win=self.losssco_win,
                      name='tes',
                      update='new')
        self.vis.line(X=epoch,
                      Y=np.array(lcc),
                      win=self.correlation_win,
                      name='lcc',
                      update='new')
        self.vis.line(X=epoch,
                      Y=np.array(srocc),
                      win=self.correlation_win,
                      name='srocc',
                      update='new')
        self.vis.scatter(X=result_list,
                         Y=np.ones(result_list.shape[0]),
                         win=self.scatter_win,
                         update='new')

        # heatmap will vertically flip the image
        senMap = np.flip(senMap, axis=0)
        img = np.flip(img, axis=0)
        errMap = np.flip(errMap, axis=0)
        self.vis.heatmap(senMap, opts=dict(title='SenMap ' + str(curr_epoch)), win=self.sen_win)
        self.vis.heatmap(img, opts=dict(title='Image ' + str(curr_epoch)), win=self.img_win)
        self.vis.heatmap(errMap, opts=dict(title='ErrMap ' + str(curr_epoch)), win=self.err_win)


def snapshot(model, testloader, epoch, best, snapshot_dir, prefix, is_first):
    val_Dict = val(model, testloader, is_first)
    lcc = val_Dict['lcc']
    srocc = val_Dict['srocc']
    test_loss = val_Dict['test_loss']

    snapshot = {
        'epoch': epoch,
        'model': model.module.state_dict(),
        'lcc': lcc,
        'srocc': srocc
    }

    if lcc + srocc >= best:
        best = lcc + srocc
        torch.save(snapshot, os.path.join(snapshot_dir, '%s_%.4f_%.4f_epoch%d.pth' %
                                          (prefix, lcc, srocc, epoch)))

    torch.save(snapshot, os.path.join(snapshot_dir, '{0}.pth'.format(prefix)))

    print("[{}] Curr LCC: {:0.4f} SROCC: {:0.4f}".format(epoch, lcc, srocc))

    out_dict = {'lcc': lcc,
                'srocc': srocc,
                'best': best,
                'test_loss': test_loss,
                'pred': val_Dict['pre_array'],
                'gt': val_Dict['gt_array'],
                'img': val_Dict['img'],
                'error': val_Dict['error'],
                'senMap': val_Dict['senMap']
                }

    return out_dict


def auxiliaryLoss(score_gt, score_pred, importance_pred):
    batch_size = score_gt.size(0)
    loss = Variable(torch.zeros(batch_size).cuda())

    for k in range(batch_size):
        _, weight = importance_pred[k].max(0)
        if weight.sum().data[0] > 0:
            final_score = score_pred[k][weight == 1].mean()
            loss[k] = nn.L1Loss()(Variable(score_gt[k].mean().data), Variable(final_score.data))
        else:
            final_score = score_pred[k].mean()
            loss[k] = nn.L1Loss()(Variable(score_gt[k].mean().data), Variable(final_score.data))

    return loss.mean()


def diceLossOnly(pred_mask, gt_mask, n_classes):
    smooth = 1.
    totalnum = gt_mask.numel()
    bsize = gt_mask.size(0)
    dice_loss = Variable(torch.zeros(bsize, n_classes).cuda())

    for c in range(0, n_classes):
        tmpmask = (gt_mask == c)
        tmplen = tmpmask.float().sum().data[0]

        if tmplen > 0:
            weights = 1 - tmplen / totalnum
        else:
            weights = 1

        # Dice loss
        for b in range(bsize):
            iflat = pred_mask[b, c, :, :]
            tflat = tmpmask[b].float()
            intersection = (iflat * tflat).sum()
            tmp = weights * (1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)))
            dice_loss[b, c] = tmp[0]

    dice_loss = dice_loss.sum(1).mean()

    return dice_loss


def totalVari_regu(senMap, beta=3):

    sobel_h = torch.Tensor([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]])
    sobel_h = sobel_h.unsqueeze(0)
    sobel_h = sobel_h.unsqueeze(0)
    sobel_h = sobel_h.cuda()
    sobel_w = torch.Tensor([[ 1,  2,  1],
                            [ 0,  0,  0],
                            [-1, -2, -1]])
    sobel_w = sobel_w.unsqueeze(0)
    sobel_w = sobel_w.unsqueeze(0)
    sobel_w = sobel_w.cuda()

    if len(senMap.shape) == 3:
        senMap = senMap.unsqueeze(1)

    h = F.conv2d(senMap, sobel_h, bias=None, stride=1, padding=1)
    w = F.conv2d(senMap, sobel_w, bias=None, stride=1, padding=1)

    tv = (h**2+w**2)**(beta/2.)

    tv = F.adaptive_avg_pool2d(tv, output_size=(1, 1)).squeeze()

    return tv


def trainProcess(model, optimG, trainloader, testloader, args, is_first):
    best_lcc = -1
    vis = visualize(args)
    # weight = torch.cuda.FloatTensor([0.5, 1.0])

    tra_loss = [3000]
    tes_loss = [3000]
    tes_lcc = [0]
    tes_srocc = [0]
    epoch_idx = [0]
    for epoch in range(1, args.max_epoch + 1):
        loss_score = []
        model.train()

        for batch_id, (img, error, score_gt) in enumerate(trainloader):
            score_gt = score_gt.type('torch.FloatTensor')
            img, error, score_gt, = Variable(img.cuda()), \
                                    Variable(error.cuda()), \
                                    Variable(score_gt.cuda())
            optimG.zero_grad()

            score_pred, senMap = model(img, error)
            # print('train', score_pred.shape, score_gt.shape)
            loss_1 = nn.MSELoss()(score_pred, score_gt)
            tv_reg = torch.mean(totalVari_regu(senMap))

            # Loss function
            LGseg = 1000*loss_1+0.01*tv_reg
            tmpsco = LGseg.data[0]

            LGseg.backward()
            itr = len(trainloader) * (epoch - 1) + batch_id
            # poly_lr_scheduler(optimG, args.lr, itr)
            optimG.step()

            loss_score.append(tmpsco)
            # loss_importance.append(tmpimp)
            print("[{0}][{1}] ScoreL1: {2:.4f} TVreg: {3:.2f}."
                  .format(epoch, itr, tmpsco, tv_reg))

        if epoch % 10 == 0:
            snap_dict = snapshot(model,
                                 testloader,
                                 epoch,
                                 best_lcc,
                                 args.snapshot_dir,
                                 args.prefix,
                                 is_first)
            lcc = snap_dict['lcc']
            srocc = snap_dict['srocc']
            best_lcc = snap_dict['best']
            test_loss = snap_dict['test_loss']
            pred_array = snap_dict['pred']
            gt_array = snap_dict['gt']

            tra_loss.append(np.array(loss_score).mean())
            tes_loss.append(test_loss)
            tes_lcc.append(lcc)
            tes_srocc.append(srocc)
            epoch_idx.append(epoch)

            sensi = snap_dict['senMap']
            img_heatmap = snap_dict['img']
            err_heatmap = snap_dict['error']

            if len(sensi.shape) != 2:
                sensi = sensi[0]
            if len(img_heatmap.shape) != 2:
                img_heatmap = img_heatmap[0]
            if len(err_heatmap.shape) != 2:
                err_heatmap = err_heatmap[0]

            # Visualize
            vis.update(losssco=tra_loss,
                       lcc=tes_lcc,
                       srocc=tes_srocc,
                       test_loss=tes_loss,
                       senMap=sensi,
                       img=img_heatmap,
                       errMap=err_heatmap,
                       epoch=epoch_idx,
                       result_list=np.array([pred_array, gt_array]))


def main():
    args = parse_args()
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    is_first = True

    # # TID2013 Dataset
    trainset = LIVEDatasetLoader(args.LIVE_dataset_dir,
                                 img_size=(112, 112),
                                 train_phase=True,
                                 is_sample=True,
                                 seed=seed,
                                 global_permute=False)
    testset = LIVEDatasetLoader(args.LIVE_dataset_dir,
                                train_phase=False,
                                is_sample=False,
                                seed=seed,
                                global_permute=False)

    train_batch_size = args.batch_size

    trainloader = DataLoader(trainset,
                             batch_size=train_batch_size,
                             shuffle=True,
                             num_workers=2,
                             drop_last=True,
                             pin_memory=True)
    testloader = DataLoader(testset,
                            shuffle=True,
                            batch_size=1,
                            num_workers=1,
                            pin_memory=True)

    model = predictNet()

    # optimG = optim.SGD(filter(lambda p: p.requires_grad, \
    #     model.parameters()),lr=args.lr,momentum=0.9,\
    #     weight_decay=1e-4,nesterov=True)
    optimG = optim.Adam(filter(lambda p: p.requires_grad,
                               model.parameters()), lr=args.lr, weight_decay=5e-3)

    model = nn.DataParallel(model).cuda()
    trainProcess(model, optimG, trainloader, testloader, args, is_first)


if __name__ == '__main__':
    main()