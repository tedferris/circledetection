from model import CNN
import torch.optim as optim
import torch.utils.data as tud
import torch
from torch.autograd import Variable
import torch.nn as nn

import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from helpers import iou, diou, diou_loss


def main():
    # Load Training Data
    x_trainlist = []
    y_temp = pd.read_csv("../data/train_set.csv")

    for filepath in y_temp['PATH']:
        x_trainlist.append(np.load("../data/" + filepath))
    
    y_trainlist = [np.array([y_temp['ROW'][i], y_temp['COL'][i], y_temp['RAD'][i]], dtype=np.float32) for i in range(len(y_temp['PATH']))]

    tensor_x = torch.tensor(np.array(x_trainlist))
    tensor_y = torch.tensor(np.array(y_trainlist))

    my_dataset = TensorDataset(tensor_x,tensor_y) # create your datset
    my_dataloader = DataLoader(my_dataset, batch_size=64, shuffle=True)

    # Initialize CNN and training parameters
    model = CNN()
    lossmetric = nn.MSELoss()
    lossmetric2 = nn.L1Loss()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    optimizer = optim.Adam(model.parameters(), lr = 0.005)

    num_epochs = 750

    for epoch in range(0, num_epochs):
        train(my_dataloader, model, lossmetric, optimizer, epoch, device, lossmetric2=lossmetric2)


def train(train_loader, model, lossmetric, optimizer, epoch, device, lossmetric2=None):
    # Set model to train
    model.train()
    running_loss = 0.0

    savedflag = 0

    #Iterate over training dataloader, converting input to FloatTensors to match bias data type
    for i, (x, y_true) in enumerate(train_loader):

        x, y_true = x.type(torch.FloatTensor), y_true.type(torch.FloatTensor)
        
        x = x.to(device)
        y_true = y_true.to(device)

        x = x.unsqueeze(1) # Reshapes input data into a format the CNN can take in.

        y_pred = model(x)

        # Calculate loss. In this case, we're using a combination of MSE and MAE.
        loss = lossmetric(y_pred, y_true) + 0.1 * lossmetric2(y_pred, y_true)
        loss = Variable(loss, requires_grad = True)


        # Compute gradient and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        # Print every 25 batches
        if i % 25 == 25 - 1 :
            new_line = 'Epoch: [%d] | loss: %f' % \
                (epoch + 1, running_loss / 25)
                    # file.write(new_line + '\n')
            print(new_line)
            running_loss = 0.0

        # Save a model checkpoint every 25 epochs.
        if epoch % 25 == 0 and not savedflag:
            print("Saving Model Checkpoint.")
            savedflag = 1
            torch.save(model.state_dict(), 'saved_models/mse_diou_combo/' + str(epoch) + '_epoch_' + 'pytorch_circle_cnn' + '_checkpoint.pth.tar')

if __name__ == '__main__':
    main()