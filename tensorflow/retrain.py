from keras.models import load_model
from keras import optimizers as opt
import pandas as pd
import numpy as np
from helpers import diou, diou_loss, iou_loss, CircleParams

# Load the previously trained model
model = load_model('models/circle_cnn_mse1_updated4.keras', custom_objects={"diou": diou, "diou_loss": diou_loss})

# Define a new learning rate
new_learning_rate = 0.008

# Recompile the model with a new optimizer and the reduced learning rate
model.compile(
    loss=iou_loss,
    optimizer=opt.Adam(learning_rate=new_learning_rate),
    metrics=['accuracy', diou],  # Using custom DIoU metric
    run_eagerly=True
)

# Load your training data
x_trainlist = []
y_temp = pd.read_csv("train_set.csv")

for filepath in y_temp['PATH']:
    x_trainlist.append(np.load(filepath))

x_train = np.stack(x_trainlist, axis=0)

y_temp.drop('PATH', axis=1, inplace=True)
y_train = y_temp.to_numpy()

# Continue training the model
model.fit(x=x_train, y=y_train, epochs=25)  # Adjust the number of epochs as needed

# Save the updated model as needed
model.save('models/circle_cnn_mse1_additional_ioutraining.keras')

print('Additional Training Completed')
