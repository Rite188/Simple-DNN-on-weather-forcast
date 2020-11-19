import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import h5py
import os
import torchvision.transforms as transforms
import time
import pandas as pd
from sklearn.model_selection import train_test_split
#pip install scikit-learn
import matplotlib.pyplot as plt
import torch.utils.data as Data

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, X,y):
        super(MyDataset, self).__init__()
        self.X = X
        self.y = y

    def __getitem__(self, index):
        X, y = torch.Tensor(self.X[index]),torch.Tensor([self.y[index]])
        return X,y

    def __len__(self):
        return len(self.X)

class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.layer1 = nn.Linear(36, 64)
        self.layer2 = nn.Linear(64,128)
        self.layer3 = nn.Linear(128, 256)
        self.layer4 = nn.Linear(256, 128)
        self.layer5 = nn.Linear(128, 1)

    def forward(self, x):
        y = self.layer1(x)
        y = F.relu(y)
        y = self.layer2(y)
        y = F.relu(y)
        y = self.layer3(y)
        y = F.relu(y)
        y = self.layer4(y)
        y = F.relu(y)
        y = self.layer5(y)
        return y

def val_plot(total_loss):
    x = range(len(total_loss))
    plt.plot(x,total_loss,label='Val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Val_loss')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('Val_loss.png')

if __name__ == "__main__":
    EPOCH = 100  # train the training data n times, to save time, we just train 1 epoch
    LR = 0.001  # learning rate
    BATCH_SIZE=10

    df = pd.read_csv('weather_data.csv').set_index('date')
    # X will be a pandas dataframe of all columns except meantempm
    X = df[[col for col in df.columns if col != 'meantempm']].values
    # Standardize X
    XMean = np.nanmean(X, axis=0)
    XStd = np.nanstd(X, axis=0)
    X = (X - XMean) / XStd

    XMin = np.nanmin(X, axis=0)
    XMax = np.nanmax(X, axis=0)
    X = (X - XMin) / (XMax - XMin)

    # y will be a pandas series of the meantempm
    y = df['meantempm'].values.astype(np.float)
    # split data into training set and a temporary set using sklearn.model_selection.traing_test_split
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.2, random_state=23)
    # take the remaining 20% of data in X_tmp, y_tmp and split them evenly
    X_test, X_val, y_test, y_val = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=23)
    print("Training instances   {}, Training features   {}".format(X_train.shape[0], X_train.shape[1]))
    print("Validation instances {}, Validation features {}".format(X_val.shape[0], X_val.shape[1]))
    print("Testing instances    {}, Testing features    {}".format(X_test.shape[0], X_test.shape[1]))

    train_data = MyDataset(X_train,y_train)
    val_data = MyDataset(X_val,y_val)
    test_data = MyDataset(X_test,y_test)

    train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = Data.DataLoader(dataset=val_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)

    time_start = time.time()
    model = DNN()
    print(model)  # net architecture
    # Loss and optimizer
    criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)#LR=0.001
    val_MSE = []
    for epoch in range(EPOCH):  # loop over the dataset multiple times
        model.train()
        train_loss = 0.0
        for step, (data, label) in enumerate(train_loader):
            output = model(data)
            loss = criterion(output, label)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if step % 10 == 9:
                print('[%d,%5d] loss: %.3f' % (epoch + 1, (step + 1)*10, train_loss / 100))
                #Batch size=10,所以每训练100个数据输出一次loss
                train_loss = 0.0
        model.eval()
        val_loss = 0.
        with torch.no_grad():  # 不需要更新模型，不需要梯度
            for step, (data, label) in enumerate(val_loader):
                output = model(data)
                loss = criterion(output, label)
                val_loss += loss.item()
            val_MSE.append(val_loss/X_val.shape[0])
        model.train()
        if len(val_MSE) == 0 or val_MSE[-1] <= min(np.array(val_MSE)):
            # 如果比之前的mse要小，就保存模型
            print("Best model on epoch: {}, val_mse: {:.4f}".format(epoch, val_MSE[-1]))
            torch.save(model.state_dict(), "Regression-best.th")
    val_plot(val_MSE)
    time_end = time.time()
    print('Training time:', time_end - time_start, 's')
    print('Train Finished')

    model = DNN()
    model.load_state_dict(torch.load('Regression-best.th'))
    with torch.no_grad():
        test_loss,test_step=0,0
        for data, label in test_loader:
            # forward
            output = model(data)
            loss = criterion(output, label)
            test_loss += loss.item()
        print("Mse of the best model on the test data is: {:.4f}".format(test_loss / X_test.shape[0]))
    print('Test Finished ')






