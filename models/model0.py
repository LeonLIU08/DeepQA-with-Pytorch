'''
The first straightforward attempt.
'''
import torch
import torch.nn as nn

torch.set_default_tensor_type('torch.FloatTensor')


class deepIQA_model(nn.Module):
    def __init__(self):
        super(deepIQA_model, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=16,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=(1, 1),
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=16,
                               out_channels=16,
                               kernel_size=(3, 3),
                               stride=(2, 2),
                               padding=(1, 1),
                               bias=True)

        self.conv3 = nn.Conv2d(in_channels=16,
                               out_channels=16,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=(1, 1),
                               bias=True)
        self.conv4 = nn.Conv2d(in_channels=16,
                               out_channels=32,
                               kernel_size=(3, 3),
                               stride=(2, 2),
                               padding=(1, 1),
                               bias=True)

        self.conv5 = nn.Conv2d(in_channels=32,
                               out_channels=32,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=(1, 1),
                               bias=True)
        self.conv6 = nn.Conv2d(in_channels=32,
                               out_channels=32,
                               kernel_size=(3, 3),
                               stride=(2, 2),
                               padding=(1, 1),
                               bias=True)

        self.relu = nn.ReLU(inplace=False)

        self.deconv1 = nn.ConvTranspose2d(32,
                                          32,
                                          kernel_size=(2, 2),
                                          stride=(2, 2),
                                          padding=(0, 0),
                                          bias=False)
        self.deconv2 = nn.ConvTranspose2d(32,
                                          16,
                                          kernel_size=(2, 2),
                                          stride=(2, 2),
                                          padding=(0, 0),
                                          bias=False)
        self.deconv3 = nn.ConvTranspose2d(16,
                                          1,
                                          kernel_size=(2, 2),
                                          stride=(2, 2),
                                          padding=(0, 0),
                                          bias=False)

        self.fc1 = nn.Linear(1, 4, bias=True)
        self.fc2 = nn.Linear(4, 1, bias=False)
        self.globalpooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, *x):
        img, error = x[0], x[1]

        x = self.relu(self.conv1(img))
        x = self.relu(self.conv2(x))

        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))

        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))

        dex = self.relu(self.deconv1(x))
        dex = self.relu(self.deconv2(dex))
        dex = self.relu(self.deconv3(dex)).squeeze()

        scoreMap = dex*error

        score = self.globalpooling(scoreMap)
        score = self.relu(self.fc1(score))
        score = self.relu(self.fc2(score)).squeeze()

        return score


if __name__ == '__main__':
    import torch
    from torch.autograd import Variable
    import time

    torch.set_num_threads(1)

    imh, imw = 384, 384
    test_image = Variable(torch.FloatTensor(15, 3, imh, imw))
    model = deepIQA_model()
    # On GPU
    model = model.cuda()
    test_image = test_image.cuda()

    # # Test BP
    # loss = nn.MSELoss()
    # pred = model(test_image)
    # output = loss(pred, Variable(torch.cuda.FloatTensor(1,20,imh,imw)))
    # output.backward()

    time_all = []
    for i in range(0, 100):
        t0 = time.clock()
        score = model(test_image, test_image[:, 0, :, :])

        t1 = (time.clock() - t0) * 1000
        fps = 1000 / t1

        print(i)
        print(score.shape)
        print(score.type())
        # print(importance.size())
        print('Forward Time: {:.2f} ms'.format(t1))
        print('Forward FPS:  {:.2f} f/s'.format(fps))
        print('')
        time_all.append(t1)

    print('Mean Time: {:.2f} ms'.format(torch.Tensor(time_all).median()))