from model import CNN
import torch.optim as optim
import torch.utils.data as tud
import torch
from torch.autograd import Variable
import torch.nn as nn

import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from helpers import CircleParams, iou


def main():
    x_trainlist = []
    y_temp = pd.read_csv("../data/train_set.csv")

    for filepath in y_temp['PATH']:
        x_trainlist.append(np.load("../data/" + filepath))
        
        # x_train = np.stack(x_trainlist, axis=0, dtype=np.float32)

    y_listofarrays = [np.array([y_temp['ROW'][i], y_temp['COL'][i], y_temp['RAD'][i]], dtype=np.float32) for i in range(len(y_temp['PATH']))]

    num_epochs = 750

    tensor_x = torch.tensor(np.array(x_trainlist)) # convert lists to numpy arrays for faster conversion to tensors
    tensor_y = torch.tensor(np.array(y_listofarrays)) # (list to tensor is a super slow process) 

    val_dataset = TensorDataset(tensor_x,tensor_y) # create your datset
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True)


    model = CNN()
    checkpoint = torch.load('saved_models/150_epoch_pytorch_circle_cnn_checkpoint.pth.tar')
    model.load_state_dict(checkpoint)

    validate(model, val_dataloader)

def validate(model, val_dataloader):
    
    model.eval()
    with torch.no_grad():

        for i, (x, y_true) in enumerate(val_dataloader):
            x, y_true = x.type(torch.FloatTensor), y_true.type(torch.FloatTensor)

            x = x.unsqueeze(1)

            y_pred = model(x)
            
            iou_list = []

            for i in range(len(y_true)):
        
                cir1 = CircleParams(y_true[i][0], y_true[i][1], y_true[i][2])
                cir2 = CircleParams(y_pred[i][0], y_pred[i][1], y_pred[i][2])

                iou_list.append(iou(cir1, cir2))
            
            print(np.mean(iou_list))

if __name__ == '__main__':
    main()            