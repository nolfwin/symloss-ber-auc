import torchvision
import torchvision.transforms as T
import torch
import torch.nn as nn
import argparse
import util
import numpy as np
import sys
from PIL import Image
from sklearn.metrics import roc_auc_score
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


class custom_cifar(torch.utils.data.Dataset):
    def __init__(self, dataset, label, transform):
        self.dataset = dataset
        self.label = label
        self.transform = T.Compose(transform)

    def __getitem__(self, index):
        _img = self.dataset[index].numpy()
        return self.transform(_img), self.label[index]

    def __len__(self):
        return len(self.dataset)


def preprocessing_cifar(root, prior, length, ind, train=True):
    length_p = int(length * prior)
    length_n = int(length - length_p)
    if train:
        data = torchvision.datasets.CIFAR10(root=root,
                                           train=True,
                                           download=True)
        img = data.train_data
        label = torch.tensor(data.train_labels, dtype=torch.float)
    else:
        data = torchvision.datasets.CIFAR10(root=root,
                                           train=False,
                                           download=True)
        img = data.test_data
        label = torch.tensor(data.test_labels, dtype=torch.float)
    x_p = img[(label == 0).nonzero()]
    x_n = img[(label == 7).nonzero()]
    ind_p = ind[0]
    ind_n = ind[1]
    x_p_o = torch.tensor(x_p[ind_p:ind_p+length_p])
    x_n_o = torch.tensor(x_n[ind_n:ind_n+length_n])
    if train:
        if prior < 0.5:
            l_p = -torch.ones(len(x_p_o))
            l_n = -torch.ones(len(x_n_o))
        else:
            l_p = torch.ones(len(x_p_o))
            l_n = torch.ones(len(x_n_o))
    else:
        l_p = torch.ones(len(x_p_o))
        l_n = -torch.ones(len(x_n_o))
    return torch.cat([x_p_o, x_n_o], dim=0), torch.cat([l_p, l_n])


class custom_mnist(torch.utils.data.Dataset):
    def __init__(self, dataset, label, transform):
        self.dataset = dataset
        self.label = label
        self.transform = T.Compose(transform)

    def __getitem__(self, index):
        _img = self.dataset[index]
        _img = _img.reshape(_img.size(1), _img.size(2))
        pad = nn.ConstantPad2d(2, 0.0)
        _img = pad(_img)
        img = Image.fromarray(_img.numpy())
        return self.transform(img), self.label[index]

    def __len__(self):
        return len(self.dataset)


def preprocessing_mnist(root, prior, length, ind, train=True):
    length_p = int(length * prior)
    length_n = int(length - length_p)
    if train:
        data = torchvision.datasets.MNIST(root=root,
                                           train=True,
                                           download=True)
        img = data.train_data.reshape(-1, 28, 28)
        label = torch.tensor(data.train_labels, dtype=torch.float)
    else:
        data = torchvision.datasets.MNIST(root=root,
                                           train=False,
                                           download=True)
        img = torch.tensor(data.test_data).reshape(-1, 28, 28)
        label = torch.tensor(data.test_labels, dtype=torch.float)
    x_p = img[(label % 2 == 0).nonzero()]
    x_n = img[(label % 2 == 1).nonzero()]
    i, j = len(x_p), len(x_n)
    i -= length_p
    j -= length_n
    ind_p = ind if ind-i < 0 else i
    ind_n = ind if ind-j < 0 else j
    x_p_o = torch.tensor(x_p[ind_p:ind_p+length_p], dtype=torch.float)
    x_n_o = torch.tensor(x_n[ind_n:ind_n+length_n], dtype=torch.float)
    l_p = torch.ones(len(x_p_o))
    l_n = -torch.ones(len(x_n_o))
    return torch.cat([x_p_o, x_n_o], dim=0), torch.cat([l_p, l_n])


batch = 128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
learning_rate = 1e-4
weight_decay = 1e-4
loss_list = ['barrier', 'sigmoid', 'unhinged', 'savage', 'log', 'sq',  'hinge']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="cifar-10", help="dataset : choose one of mnist or cifar-10. Note that cifar-10 is Airplane vs Horse")
    parser.add_argument("--epoch", type=int, default=50, help="the number of epoch to train model")
    parser.add_argument("--prior", type=float, default=0.65, help="Class prior for corrupted label. 1.0 for (1.0, 0.0), 0.8 for (0.8, 0.3), 0.7 for (0.7, 0.4) and 0.65 for (0.65, 0.45)")
    parser.add_argument("--opt", type=str, default="auc", help="ber or auc optimization")

    config = parser.parse_args()
    assert config.data in ['mnist','cifar-10'], 'dataset ERROR : choose mnist or cifar-10'
    assert config.opt in ['ber', 'auc'], 'Optimization ERROR : choose ber or auc'

    if config.prior == 1.0:
        pi_pos = 1.0
        pi_neg = 0.0
    else :
        pi_pos = config.prior
        pi_neg = 1.1 - pi_pos
    print('Data: %s' % config.data)
    score = np.zeros(shape=(7))
    if config.data == 'mnist':
        img_size = [1, 32, 32]
        transform = [T.ToTensor()]
        d, l = preprocessing_mnist(root='../../data', prior=0.5, length=9800, ind=0,
                             train=False)
        test_data = custom_mnist(dataset=d, label=l, transform=transform)
        input_dim = img_size[0] * img_size[1] * img_size[2]
        d, l = preprocessing_mnist(root='../../data', prior=pi_pos, length=15000, ind=0)
        class_1 = custom_mnist(dataset=d, label=l, transform=transform)
        d, l = preprocessing_mnist(root='../../data', prior=pi_neg, length=15000, ind=15000)
        class_2 = custom_mnist(dataset=d, label=l, transform=transform)

        for type, loss in enumerate(loss_list):
            model = util.Cor(loss=loss, size=img_size).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True)
            class_1_loader = torch.utils.data.DataLoader(dataset=class_1,
                                                         batch_size=batch,
                                                         shuffle=True)
            class_2_loader = torch.utils.data.DataLoader(dataset=class_2,
                                                         batch_size=batch,
                                                         shuffle=True)
            test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                                      batch_size=batch,
                                                      shuffle=True)
            for epoch in range(config.epoch):
                print('%s, epoch: %s/%s' % (loss, epoch+1, config.epoch))
                for i, ((img1, _), (img2, __)) in enumerate(zip(class_1_loader, class_2_loader)):
                    image_s = img1.reshape(-1, img_size[0], img_size[1], img_size[2]).to(device)
                    image_t = img2.reshape(-1, img_size[0], img_size[1], img_size[2]).to(device)
                    err = model(image_s, image_t, mode=config.opt)
                    optimizer.zero_grad()
                    err.backward()
                    optimizer.step()
            if config.opt == 'ber':
                sc = 0
                for i, (img, label) in enumerate(test_loader):
                    image_t = img.reshape(-1, img_size[0], img_size[1], img_size[2]).to(device)
                    label_t = label.reshape(-1, 1)
                    pred = torch.sign(model.forward_test(image_t))
                    sc += (pred - label_t.to(device)).nonzero().size(0)
                sc = 1-sc/len(test_data)
            elif config.opt =='auc':
                sc = list()
                for i, (img, label) in enumerate(test_loader):
                    image_t = img.reshape(-1, img_size[0], img_size[1], img_size[2]).to(device)
                    label_t = label.reshape(-1)
                    pred = model.forward_test(image_t)
                    sc.append(roc_auc_score(label_t, pred.cpu().detach().numpy()))
                sc = np.mean(np.asarray(sc))
            score[type] = 100*sc

    elif config.data =='cifar-10':
        img_size = [3, 32, 32]
        transform = [T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))]
        d, l = preprocessing_cifar(root='../../data', prior=0.5, length=2000, ind=[0, 0],
                             train=False)
        test_data = custom_cifar(dataset=torch.squeeze(d, dim=1), label=l, transform=transform)
        input_dim = img_size[0] * img_size[1] * img_size[2]
        d_1, l_1 = preprocessing_cifar(root='../../data', prior=pi_pos, length=4540, ind=[0, 0])
        d_2, l_2 = preprocessing_cifar(root='../../data', prior=pi_neg, length=4540,
                                 ind=[int(4540 * pi_pos) + 1, int(4540 * (1 - pi_pos)) + 1])
        d = torch.cat([torch.squeeze(d_1, dim=1), torch.squeeze(d_2, dim=1)])
        l = torch.cat([l_1, l_2])
        train_data = custom_cifar(dataset=d, label=l, transform=transform)
        class_1_loader = torch.utils.data.DataLoader(dataset=train_data,
                                                     batch_size=batch,
                                                     shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                                  batch_size=batch,
                                                  shuffle=True)
        for type, loss in enumerate(loss_list):
            model = util.Cor(loss=loss, size=img_size).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True, weight_decay=weight_decay)

            for epoch in range(config.epoch):
                print('%s, epoch: %s/%s' % (loss, epoch+1, config.epoch))
                for i, (img, label) in enumerate(class_1_loader):
                    image_p = img[label == 1].to(device)
                    image_n = img[label == -1].to(device)
                    err = model(image_p, image_n, mode=config.opt)
                    optimizer.zero_grad()
                    err.backward()r
                    optimizer.step()

            if config.opt == 'ber':
                sc = 0
                for i, (img, label) in enumerate(test_loader):
                    image_t = img.reshape(-1, img_size[0], img_size[1], img_size[2]).to(device)
                    label_t = label.reshape(-1, 1)
                    pred = torch.sign(model.forward_test(image_t))
                    sc += (pred - label_t.to(device)).nonzero().size(0)
                sc = 1-sc/len(test_data)
            elif config.opt =='auc':
                sc = list()
                for i, (img, label) in enumerate(test_loader):
                    image_t = img.reshape(-1, img_size[0], img_size[1], img_size[2]).to(device)
                    label_t = label.reshape(-1)
                    pred = model.forward_test(image_t)
                    sc.append(roc_auc_score(label_t, pred.cpu().detach().numpy()))
                sc = np.mean(np.asarray(sc))
            score[type] = 100*sc

    print(config.data, ' ', config.opt, ' optimization score where ', r"[\pi, \pi']= ", '[', pi_pos,',', pi_neg, ']')
    for i in range(len(loss_list)):
        print(loss_list[i], ' : ', score[i])
