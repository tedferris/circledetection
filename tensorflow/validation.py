from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import optimizers
from keras.metrics import Accuracy

from helpers import diou, diou_loss, iou, CircleParams

import pandas as pd
import numpy as np
import keras

# Load model
train_results_path = "datasets/validationset/"
model_path = "circle_cnn_mse1_updated4.keras"

model = keras.models.load_model(model_path, compile=False)

# Load validation dataset and reshape
x_list = []
y_temp = pd.read_csv("validation_set.csv")

for filepath in y_temp['PATH']:
    x_list.append(np.load(filepath))

x_validation = np.stack(x_list, axis=0)

# Convert ground truths to list of CircleParams
y_params = [CircleParams(y_temp['ROW'][i], y_temp['COL'][i], y_temp['RAD'][i]) for i in range(len(y_temp['PATH']))]


# Run model predictions over validation dataset, convert results to list of CircleParams
y_preds = model.predict(x_validation)

y_preds_params = [CircleParams(y_preds[i][0], y_preds[i][1], y_preds[i][2]) for i in range(len(y_preds))]

# Calculate IoU and DIoU over validation set
iou_validation = [iou(y_preds_params[i], y_params[i]) for i in range(len(y_params))]
diou_validation = [diou(y_preds_params[i], y_params[i]) for i in range(len(y_params))]

print(f"MEAN IOU: {np.mean(iou_validation)}")
print(f"MEAN DIOU: {np.mean(diou_validation)}")
