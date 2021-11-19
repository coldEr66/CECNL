import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as Data
from scipy import io
from torch.utils.tensorboard import SummaryWriter

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
        self.conv1 = nn.Conv2d(1, 22, (22, 1))
        self.Bn1 = nn.BatchNorm2d(22)
        self.conv2 = nn.Conv2d(22, 20, (1, 12), padding=(0, 6))
        self.Bn2   = nn.BatchNorm2d(20)
        self.SquareLayer = square_layer()
        self.Drop1 = nn.Dropout(0.6)
        self.AvgPool1 = nn.AvgPool2d((1, 62), stride=(1, 12))
        self.log_layer = log_layer()
        self.classifier = nn.Linear(840, 4, bias=True)
        # self.softmax = nn.Softmax()
    
    def forward(self, x):
        # print(type(x))
        l = x[:, 0, :, :, :]
        r = x[:, 1, :, :, :]
        # print(type(l),type(r))
        # print(l.size(), r.size())
        # print(x.size())
        l, r = self.conv1(l), self.conv1(r)
        out = l + r
        out = self.Bn1(out)
        out = self.Drop1(out)

        out = self.conv2(out)
        out = self.Bn2(out)

        out = self.SquareLayer(out)
        out = self.Drop1(out)

        out = self.AvgPool1(out)
        out = self.log_layer(out)

        out = out.view(-1, 840)
        out = self.classifier(out)
        return out

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
        
    tmp_X = np.array(tmp)
    tmp_y = []
    for i in range(4):
        tp = []
        for j in range(72):
            tp.append(i)
        np.random.shuffle(tmp_X[i])
        tmp_y.append(tp)
    tmp_y = np.array(tmp_y)

    train_X = np.concatenate((tmp_X[0, :60], tmp_X[1, :60], tmp_X[2, :60], tmp_X[3, :60]), axis = 0)
    train_y = np.concatenate((tmp_y[0, :60], tmp_y[1, :60], tmp_y[2, :60], tmp_y[3, :60]), axis = 0)
    print(train_X.shape)
    print(train_y.shape)

    validate_X = np.concatenate((tmp_X[0, 60:], tmp_X[1, 60:], tmp_X[2, 60:], tmp_X[3, 60:]), axis = 0)
    validate_y = np.concatenate((tmp_y[0, 60:], tmp_y[1, 60:], tmp_y[2, 60:], tmp_y[3, 60:]), axis = 0)



    x_train_list = []
    y_train_list = []
    for i in range(train_X.shape[0]):
        for j in range(train_X.shape[0]):
            if train_y[i] != train_y[j]:
                continue
            x_train_list.append((train_X[i], train_X[j]))
            y_train_list.append(train_y[i])
            # y_train_list.append(train_data_tmp['y_train'][i] * 4 + train_data_tmp['y_train'][j])

    x_train_data = np.array(x_train_list)
    y_train_data = np.array(y_train_list)

    x_validate_list = []
    y_validate_list = []
    for i in range(validate_X.shape[0]):
        x_validate_list.append((validate_X[i], validate_X[i]))
        y_validate_list.append(validate_y[i])

    x_validate_data = np.array(x_validate_list)
    y_validate_data = np.array(y_validate_list)

    
    x_test_list = []
    y_test_list = []
    for i in range(test_data_tmp['x_test'].shape[0]):
        x_test_list.append((test_data_tmp['x_test'][i], test_data_tmp['x_test'][i]))
        y_test_list.append(test_data_tmp['y_test'][i])

        # for j in range(test_data_tmp['y_test'].shape[0]):
            # if test_data_tmp['y_test'][i] != test_data_tmp['y_test'][j]:
                # continue
            # if i != j
                # continue
            # x_test_list.append((test_data_tmp['x_test'][i], test_data_tmp['x_test'][j]))
            # y_test_list.append(test_data_tmp['y_test'][j])
            # y_test_list.append(test_data_tmp['y_test'][i] * 4 + test_data_tmp['y_test'][j])
    
    x_test_data = np.array(x_test_list)
    y_test_data = np.array(y_test_list)
    
    # print(x_test_data.shape)

    # change shape to [BatchSize, channel, EEGchannel, time]
    x_train = torch.Tensor(x_train_data).unsqueeze(2)
    x_validate = torch.Tensor(x_validate_data).unsqueeze(2)
    x_test = torch.Tensor(x_test_data).unsqueeze(2)
    # change type to long for calculating loss function later
    y_train = torch.Tensor(y_train_data).view(-1).long()
    y_validate = torch.Tensor(y_validate_data).view(-1).long()
    y_test = torch.Tensor(y_test_data).view(-1).long()

    print(f"x_train size is: {x_train.size()}")
    print(f"y_train size is: {y_train.size()}")
    print(f"x_validate size is: {x_validate.size()}")
    print(f"y_validate size is: {y_validate.size()}")
    print(f"x_test size is: {x_test.size()}")
    print(f"y_test size is: {y_test.size()}")

    if torch.cuda.is_available():
        dev = torch.device("cuda")
        num_worker = 0
        x_train = x_train.to(dev)
        y_train = y_train.to(dev)
        x_validate = x_validate.to(dev)
        y_validate = y_validate.to(dev)
        x_test = x_test.to(dev)
        y_test = y_test.to(dev)
    else:
        print("cpu")
        dev = torch.device("cpu")
        num_worker = 8

    train_dataset = Data.TensorDataset(x_train, y_train)
    validate_dataset = Data.TensorDataset(x_validate, y_validate)
    test_dataset = Data.TensorDataset(x_test, y_test)

    small_train_dataset, remain_train_dataset = Data.random_split(train_dataset, [288,len(train_dataset) - 288])
    small_test_dataset, remain_test_dataset = Data.random_split(test_dataset, [288,len(test_dataset) - 288])
    # print(len(small_train_dataset))

    trainloader = Data.DataLoader(
        # dataset = small_train_dataset,
        dataset = train_dataset,
        batch_size = 4096,
        shuffle = True,
        num_workers = num_worker,
    )

    validateloader = Data.DataLoader(
        dataset = validate_dataset,
        batch_size = 1,
        shuffle = True,
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

def train(net, train_dataloader, validate_dataloader, subject, num, write, epochs=3000):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    
    print('\nstart training')
    net.train()
    name = f"subject_{subject}"
    num = f"num_{num}_lr-4_bs4096_drp6x2_0"
    if write:
        writer = SummaryWriter("experiment/july31/" + name + '/' + num + '/')
    for epoch in range(epochs):
        for xb, yb in train_dataloader:
            pred = net(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        v_acc = 0
        for xb, yb in validate_dataloader:
            pred = net(xb)
            if pred.argmax() == yb:
                v_acc += 1
            v_loss = criterion(pred, yb)
        v_acc /= len(validate_dataloader)
        if write:
            writer.add_scalar("Loss/train", loss.item(), epoch)
            writer.add_scalar("Loss/validation", v_loss.item(), epoch)
            writer.add_scalar("accuracy/validation", v_acc, epoch)

        # if (epoch + 1) % 2 == 0:
            # print(f"epoch {epoch + 1} validation loss: {v_loss.item()}")
            # print(f"epoch {epoch + 1} validation accuracy: {v_acc}")
        if (epoch + 1) % 10 == 0:
            print(f"epoch {epoch + 1} loss: {loss.item()}")
    if write:
        writer.close()
    print('saving model')
    torch.save(net, 'save_net/single_net_' + str(subject) + '.pt')


def test(net, testloader):
    print('\nstart testing')
    acc = 0
    net.eval()
    for x, y in testloader:
        pred = net(x)
        if pred.argmax() == y:
            acc += 1
    acc /= len(testloader)
    print(f"accuracy: {acc}")
    return acc

def main():
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    num = 10
    acc_list = []
    for i in range(1, 10):
        acc = 0.0
        for j in range(num):
            dsccnet = DSCCNet().to(dev)
            print("The result of subject " + str(i))
            trainloader, validateloader, testloader = getDataLoader(i)
            
            train(dsccnet, trainloader, validateloader, i, j, 0)
            acc += test(dsccnet, testloader)
        
        acc /= num
        acc_list.append(acc)

        # print(dsccnet)
    for i in range(len(acc_list)):
        print(f"accuracy of subject {i+1}: {acc_list[i]}")
    acclist = np.array(acc_list)
    print(f"The mean of the accuracy is {np.mean(acclist)}")
    # trainloader, testloader = getDataLoader()
    # net = torch.load('add_div2_net.pt')
    # test(net, testloader)

if __name__ == '__main__':
    main()
    
