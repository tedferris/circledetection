from model import CNN
import torch.optim as optim
import torch
from torch.autograd import Variable
import torch.nn as nn

import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from helpers import diou_loss

def main():
    # Load Training Data
    x_trainlist = []
    y_temp = pd.read_csv("../data/train_set.csv")

    for filepath in y_temp['PATH']:
        x_trainlist.append(np.load("../data/" + filepath))
        
    y_trainlist = [np.array([y_temp['ROW'][i], y_temp['COL'][i], y_temp['RAD'][i]], dtype=np.float32) for i in range(len(y_temp['PATH']))]

    tensor_x = torch.tensor(np.array(x_trainlist)) # convert lists to numpy arrays for faster conversion to tensors
    tensor_y = torch.tensor(np.array(y_trainlist)) # (list to tensor is a super slow process) 

    train_dataset = TensorDataset(tensor_x,tensor_y) # create tensor datset
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Initialize CNN, load in saved checkpoint and define training metrics/parameters
    model = CNN()
    checkpoint = torch.load('saved_models/pure_mse/150_epoch_pytorch_circle_cnn_checkpoint.pth.tar')
    model.load_state_dict(checkpoint)

    lossmetric = diou_loss
    lossmetric2 = nn.MSELoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_epochs = 75
    optimizer = optim.Adam(model.parameters(), lr = 0.005)

    # Run the retraining loop
    for epoch in range(0, num_epochs):
        train(train_dataloader, model, lossmetric, optimizer, epoch, device, lossmetric2)


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

        # Calculate loss. In this case, we're using a combination of DIoU and MSE.
        loss = lossmetric(y_pred, y_true) + 0.1 * lossmetric2(y_pred, y_true)
        loss = Variable(loss, requires_grad = True)

        # Compute gradient and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update running loss value
        running_loss += loss.item()
        
        # Print every 10 batches
        if i % 10 == 10 - 1 : 
            new_line = f'Epoch: {epoch + 1} | loss: {running_loss / 25}'
            print(new_line)
            running_loss = 0.0

        # Save a model checkpoint every 25 epochs
        if epoch % 25 == 0 and not savedflag:
            print("Saving Model Checkpoint.")
            savedflag = 1
            torch.save(model.state_dict(), 'saved_models/mse_diou_combo/' + str(epoch) + '_epoch_' + 'pytorch_circle_cnn' + '_checkpoint.pth.tar')

if __name__ == '__main__':
    main()