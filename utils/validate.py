import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import scipy.stats


def val(model, testloader, is_first=True):
    model.eval()
    results = []
    test_loss = []

    for img_id, (img, error, score_gt) in enumerate(testloader):
        score_gt = score_gt.type('torch.FloatTensor')
        img, error = Variable(img.cuda()), Variable(error.cuda())
        # score_gt = score_gt.cuda()

        score_pred, senMap = model(img, error)

        score_pred = score_pred.data.cpu()

        for k in range(score_gt.size(0)):

            final_score = score_pred[k]
            # print(score_gt[k], final_score)
            results.append([score_gt[k], final_score])

            loss = 1000*nn.MSELoss()(score_gt[k], final_score)
            test_loss.append(loss)

    results = np.array(results)
    lcc = np.corrcoef(results, rowvar=False)[0][1]
    srocc = scipy.stats.spearmanr(results[:, 0], results[:, 1])[0]

    img = img.squeeze()
    error = error.squeeze()
    senMap = senMap.squeeze()
    outdict = {'lcc': lcc,
               'srocc': srocc,
               'test_loss': np.mean(test_loss),
               'pre_array': results[:, 0],
               'gt_array': results[:, 1],
               'img': img.data.cpu().numpy(),
               'error': error.data.cpu().numpy(),
               'senMap': senMap.data.cpu().numpy()}

    return outdict
