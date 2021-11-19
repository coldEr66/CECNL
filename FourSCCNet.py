import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as Data
from torchvision import transforms
from scipy import io
import collections
import random


dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
num_x = 5 # number of references
# you can define your own layer
class square_layer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x ** 2

class log_layer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.log(x)

class SCCNet(nn.Module):
    def __init__(self):
        super(SCCNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 22, (22, 1))
        self.Bn1 = nn.BatchNorm2d(22)
        self.conv2 = nn.Conv2d(22, 20, (1, 12), padding=(0, 6))
        self.Bn2   = nn.BatchNorm2d(20)
        self.SquareLayer = square_layer()
        self.Drop1 = nn.Dropout(0.5)
        self.AvgPool1 = nn.AvgPool2d((1, 62), stride=(1, 12))
        self.log_layer = log_layer()
        self.classifier = nn.Linear(840, 4, bias=True)
        # self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.conv1(x)
        x = self.Bn1(x)
        x = self.conv2(x)
        x = self.Bn2(x)
        x = self.SquareLayer(x)
        x = self.Drop1(x)
        x = self.AvgPool1(x)
        x = self.log_layer(x)
        x = x.view(-1, 840)
        x = self.classifier(x)
        # x = self.softmax(x)
        return x

class DSCCNet(nn.Module):
    def __init__(self):
        super(DSCCNet, self).__init__()
        self.conv1_1 = nn.Conv2d(1, 22, (22, 1))
        self.conv1_2 = nn.Conv2d(1, 22, (22, 1))
        self.Bn1_1 = nn.BatchNorm3d(22)
        self.Bn1_2 = nn.BatchNorm3d(22)
        self.conv2_1 = nn.Conv3d(22, 20, (2, 1, 12), padding=(0, 0, 6))
        self.conv2_2 = nn.Conv3d(22, 20, (2, 1, 12), padding=(0, 0, 6))
        self.Bn2_1 = nn.BatchNorm3d(20)
        self.Bn2_2 = nn.BatchNorm3d(20)
        self.SquareLayer = square_layer()
        self.Drop1 = nn.Dropout(0.5)

        self.AvgPool1_1 = nn.AvgPool3d((2, 1, 62), stride=(1, 1, 12))
        self.AvgPool1_2 = nn.AvgPool3d((2, 1, 62), stride=(1, 1, 12))

        # self.AvgPool1_1 = nn.AvgPool3d((1, 1, 62), stride=(1, 1, 12))
        # self.AvgPool1_2 = nn.AvgPool3d((1, 1, 62), stride=(1, 1, 12))

        self.log_layer = log_layer()
        self.classifier = nn.Linear(1680, 4, bias=True)
        # self.classifier = nn.Linear(3360, 4, bias=True)
    
    def forward(self, x):
        l = x[:, 0, :, :, :]
        r = x[:, 1, :, :, :]
        l, r = self.conv1_1(l), self.conv1_1(r)
        l = torch.unsqueeze(l, 2)
        r = torch.unsqueeze(r, 2)
        out_l = torch.cat((l, r), 2)
        out_r = torch.cat((r, l), 2) # [batchsize, stack = 2, 22, 1, 562]
        l, r = self.Bn1_1(out_l), self.Bn1_2(out_r)
        # l, r = self.Bn1_1(out_l), self.Bn1_1(out_r)
        
        l, r = self.conv2_1(l), self.conv2_2(r)
        # l, r = self.conv2_1(l), self.conv2_1(r)
        out_l = torch.cat((l, r), 2)
        out_r = torch.cat((r, l), 2)
        l, r = self.Bn2_1(out_l), self.Bn2_2(out_r)
        # l, r = self.Bn2_1(out_l), self.Bn2_1(out_r)

        l, r = self.SquareLayer(l), self.SquareLayer(r)
        l, r = self.Drop1(l), self.Drop1(r)
        l, r = self.AvgPool1_1(l), self.AvgPool1_2(r)
        # l, r = self.AvgPool1_1(l), self.AvgPool1_1(r)

        l, r = self.log_layer(l), self.log_layer(r)
        out = torch.cat((l, r), 2)

        out = out.view(-1, 1680)
        # out = out.view(-1, 3360)
        # res = self.classifier(out)
        return out

class FourDSCCNet(nn.Module):
    def __init__(self):
        super(FourDSCCNet, self).__init__()
        self.conv1 = nn.ModuleList()
        self.Bn1 = nn.ModuleList() 
        self.conv2 = nn.ModuleList()
        self.Bn2 = nn.ModuleList()
        self.Drop = nn.ModuleList()
        self.AvgPool = nn.ModuleList()
        for i in range(8):
            self.conv1.append(nn.Conv2d(1, 22, (22, 1)))
            self.Bn1.append(nn.BatchNorm3d(22))
            self.conv2.append(nn.Conv3d(22, 20, (2, 1, 12), padding=(0, 0, 6)))
            self.Bn2.append(nn.BatchNorm3d(20))
            self.Drop.append(nn.Dropout(0.5))
            self.AvgPool.append(nn.AvgPool3d((2, 1, 62), stride=(1, 1, 12)))

        self.SquareLayer = square_layer()
        self.log_layer = log_layer()

        self.classifier = nn.Linear(6720, 4, bias = True)

    def forward(self, x):
        out = []
        for i in range(0, 8, 2):
            l = x[:, i, :, :, :]
            r = x[:, i+1, :, :, :]
            l, r = self.conv1[i](l), self.conv1[i+1](r)
            l = torch.unsqueeze(l, 2)
            r = torch.unsqueeze(r, 2)
            out_l = torch.cat((l, r), 2)
            out_r = torch.cat((r, l), 2) # [batchsize, stack = 2, 22, 1, 562]

            l, r = self.Bn1[i](out_l), self.Bn1[i+1](out_r)
            
            l, r = self.conv2[i](l), self.conv2[i+1](r)
            out_l = torch.cat((l, r), 2)
            out_r = torch.cat((r, l), 2)
            l, r = self.Bn2[i](out_l), self.Bn2[i+1](out_r)

            l, r = self.SquareLayer(l), self.SquareLayer(r)
            l, r = self.Drop[i](l), self.Drop[i+1](r)
            l, r = self.AvgPool[i](l), self.AvgPool[i+1](r)

            l, r = self.log_layer(l), self.log_layer(r)
            tmp = torch.cat((l, r), 2)
            
            tmp = tmp.view(-1, 1680)
            out.append(tmp)
        
        res = torch.cat((out[0], out[1], out[2], out[3]), 1)
        res = self.classifier(res)
        return res


class Dataset(Data.Dataset):
    def __init__(self, dataX, datay, ref, x, transform = None): # size of references are x * 4
        self.dict = collections.defaultdict(list)
        self.ref = ref
        self.dataX = dataX
        self.datay = datay
        self.x = x
        self.transform = transform

        for i in range(x ** 4):
            cur = []
            tmp = i
            for j in range(4):
                cur.append(tmp % x)
                tmp = tmp // x
            self.dict[i] = cur

    def __len__(self):
        return (self.x ** 4) * self.dataX.shape[0]

    def __getitem__(self, idx):
        index, index_dic = idx // (self.x ** 4), idx % (self.x ** 4)
        x = self.dataX[index]
        lst = self.dict[index_dic]
        L, R, F, T = self.ref[0][lst[0]], self.ref[1][lst[1]], self.ref[2][lst[2]], self.ref[3][lst[3]]

        X_tmp = np.concatenate(([L], [x], [R], [x], [F], [x], [T], [x]), axis = 0)
        y_tmp = [self.datay[index]]
        
        X = torch.Tensor(X_tmp).unsqueeze(1)
        y = torch.Tensor(y_tmp).view(-1).long()
        y = torch.squeeze(y)

        return (X, y)

    


def getDataLoader(subject):
    filename = '../dataset/' + 'BCIC_S0' + str(subject)
    train_data_tmp = io.loadmat(filename + '_T.mat')
    test_data_tmp = io.loadmat(filename + '_E.mat')

    """
    print(train_data_tmp['x_train'].shape)
    print(train_data_tmp['y_train'].shape)
    print(test_data_tmp['x_test'].shape)
    print(test_data_tmp['y_test'].shape)
    """
    

    tmp = [[], [], [], []]
    for i in range(train_data_tmp['x_train'].shape[0]):
        idx = train_data_tmp['y_train'][i][0]
        tmp[idx].append(train_data_tmp['x_train'][i])

    Lref = np.array([random.sample(tmp[0], num_x)])
    Rref = np.array([random.sample(tmp[1], num_x)])
    Fref = np.array([random.sample(tmp[2], num_x)])
    Tref = np.array([random.sample(tmp[3], num_x)])
    ref = np.concatenate((Lref, Rref, Fref, Tref), axis = 0)

    tmp_X = np.array(tmp)
    tmp_y = []
    for i in range(4):
        tp = []
        for j in range(72):
            tp.append(i)
        np.random.shuffle(tmp_X[i])
        tmp_y.append(tp)
    tmp_y = np.array(tmp_y)

    train_X = np.concatenate((tmp_X[0, :54], tmp_X[1, :54], tmp_X[2, :54], tmp_X[3, :54]), axis = 0)
    train_y = np.concatenate((tmp_y[0, :54], tmp_y[1, :54], tmp_y[2, :54], tmp_y[3, :54]), axis = 0)

    validate_X = np.concatenate((tmp_X[0, 54:], tmp_X[1, 54:], tmp_X[2, 54:], tmp_X[3, 54:]), axis = 0)
    validate_y = np.concatenate((tmp_y[0, 54:], tmp_y[1, 54:], tmp_y[2, 54:], tmp_y[3, 54:]), axis = 0)

    test_X, test_y = [], []
    for i in range(test_data_tmp['x_test'].shape[0]):
        test_X.append(test_data_tmp['x_test'][i])
        test_y.append(test_data_tmp['y_test'][i])
    test_X, test_y = np.array(test_X), np.array(test_y)

    
    train_dataset = Dataset(train_X, train_y, ref, num_x)
    validate_dataset = Dataset(validate_X, validate_y, ref, num_x)
    test_dataset = Dataset(test_X, test_y, ref, num_x)

    if torch.cuda.is_available():
        dev = torch.device("cuda")
        num_worker = 0
    else:
        print("cpu")
        dev = torch.device("cpu")
        num_worker = 8

    trainloader = Data.DataLoader(
        # dataset = small_train_dataset,
        dataset = train_dataset,
        batch_size = 128,
        shuffle = True,
        num_workers = num_worker,
    )
    
    validateloader = Data.DataLoader(
        dataset = validate_dataset,
        batch_size = 1,
        shuffle = False,
        num_workers = num_worker,
    )

    testloader =  Data.DataLoader(
        # dataset = small_test_dataset,
        dataset = test_dataset,
        batch_size = 1,
        shuffle = False,
        num_workers = num_worker,
    )

    return trainloader, validateloader, testloader

def train(net, train_dataloader, validate_dataloader, subject, epochs=20):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-3)

    print('\nstart training')
    net.train()
    for epoch in range(epochs):
        for xb, yb in train_dataloader:
            xb, yb = xb.to(dev), yb.to(dev)
            pred = net(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        res = []
        vote_valid_acc = 0
        valid_acc = 0

        for i, (xb, yb) in enumerate(validate_dataloader):
            xb, yb = xb.to(dev), yb.to(dev)
            tmp = net(xb)

            # 1458
            if tmp.argmax() == yb:
                valid_acc += 1
            res.append(tmp)

            if (i + 1) % (num_x ** 4) == 0:
                res = torch.stack(res)
                pred = torch.mean(res, 0)
                if pred.argmax() == yb:
                    vote_valid_acc += 1

                valid_loss = criterion(pred, yb)
                # valid_loss.backward()
                # optimizer.step()
                # optimizer.zero_grad()
                res = []
        vote_valid_acc /= (len(validate_dataloader) / (num_x ** 4))
        valid_acc /= len(validate_dataloader)

        if (epoch + 1) % 5 == 0:
            print(f"epoch {epoch+1} validation loss: {valid_loss.item()}")
            print(f"epoch {epoch+1} voted validation accuracy: {vote_valid_acc}")
            print(f"epoch {epoch+1} validation accuracy: {valid_acc}")

        if (epoch + 1) % 10 == 0:
            print(f"epoch {epoch+1} loss: {loss.item()}")
    
    # print('saving model')
    # torch.save(net, 'save_net/Four_net_' + str(subject) + '.pt')


def test(net, testloader):
    print('\nstart testing')
    acc = 0
    res = []
    net.eval()
    for i, (x, y) in enumerate(testloader):
        x, y = x.to(dev), y.to(dev)
        tmp = net(x)
        res.append(tmp)

        if (i + 1) % (num_x ** 4) == 0:
            res = torch.stack(res)
            pred = torch.mean(res, 0)
            if pred.argmax() == y:
                acc += 1
            res = []
    print(len(testloader) / (num_x ** 4))
    acc /= (len(testloader) / (num_x ** 4))
    print(f"accuracy: {acc}")
    return acc

def main():
    
    num_of_train = 1 
    acc_list = []
    for i in range(1, 2):
        acc = 0
        for j in range(num_of_train):
            FDsccnet = FourDSCCNet().to(dev)
            print(FDsccnet)
            print("The result of subject " + str(i))
            trainloader, validateloader, testloader = getDataLoader(i)

            train(FDsccnet, trainloader, validateloader, i)
            acc += test(FDsccnet, testloader)
        acc /= num_of_train
        acc_list.append(acc)

    for i in range(len(acc_list)):
        print(f"accuracy of subject {i+1}: {acc_list[i]}")
    acclist = np.array(acc_list)
    print(f"The mean of the accuracy is {np.mean(acclist)}")
    # trainloader, testloader = getDataLoader()
    # net = torch.load('add_div2_net.pt')
    # test(net, testloader)

if __name__ == '__main__':
    main()
    
