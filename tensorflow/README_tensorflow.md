This directory contains all the progress made towards training a Circle Detection CNN in Tensorflow. The structure is as follows:

**helpers.py** - Contains the given helper functions, as well as five additional functions to aid in data processing and model training. The additional functions are slight modifications to the given functions:
- diou() is an extension of the given iou function, incorporating an additional factor in the calculation. Rather than returning a value between 0 and 1 purely based on overlap, diou returns iou minus a distance factor, measuring how far the centre of the predicted circle is from ground truth. The final function return takes the following form:
    - DIoU = IoU - (d^2/C^2), where d is the distance between the centres of each circle, and C is the diameter of the bounding circle (the smallest circle that fully contains both the predicted circle and ground truth). In contrast to traditional IoU, which has range [0, 1], DIoU has range [-1, 1], where the worst-case is two non-overlapping circles of radius 0.
- backend_diou() contains the same logic as diou(), but is built to take in two Tensors, rather than two CircleParam objects. The logic remains the same, but DIoU is calculated for every entry in the tensor, and the mean of the calculated DIoU values is returned.
- diou_loss() is a loss function that implements backend_diou(), normalizing the result from [-1, 1] to [0, 1] and returning 1 - diou. This gives us a standard loss function with range [0, 1] that we can train with.
- backend_iou() contains the given IoU logic, again translated to work with Tensors. This enables us to use the IoU logic in model training.
- iou_loss() is similar to diou_loss(), but implements backend_iou() rather than backend_diou()


**train.py** - The script used for model creation and training. A pretty standard CNN architecture was used, with 4 convolutional layers and a fully connected layer at the end to flatten and linearize the outputs. After creating and compiling the model using an Adam optimizer, the training data is loaded and some light processing is done to get the data into a trainable state. The CNN is then trained and saved to the models folder.

**retrain.py** - This script is used to load a keras model, continue training, and save the model again. This was useful for conducting some split-training experiments, where a CNN would be trained to okay results using MSE or another standard loss function, then trained from that point using custom IoU or DIoU loss.

**validation.py** - Used to validate model results. This script loads a saved keras model, generates predictions over the validation set, and uses the given IoU function to compute the mean IoU for the entire validation set. 


**Experiments and Learnings:**

A number of different tests were run to try and get a robust and effective CNN. Initial training with MSE and MAE wasn't the most successful, and it seemed intuitive to train a model using a loss function more closely resembling the IoU metric used to validate its predictions, so I searched (unfruitfully) online to find a TensorFlow iou loss implementation. Ultimately, I decided that it made the most sense to translate the given loss function to intake Tensors, and modify some of the logic to work entirely in TensorFlow's backend. 

This ended up being a more involved process than I was expecting, and led to a major problem - in the early epochs of training, the CNN would have a number of guesses with no overlap. This meant IoU was 0, and loss was constant at 1. This created a problem in backpropagation, as there was no gradient to be computed and descended through. This realization, along with coming across this paper from Cornell (https://arxiv.org/abs/1911.08287), led me towards DIoU. The added distance factor rewards the model for guessing that are closer to ground truth, even when there is no overlap, and creates a fully differentiable function.

However, even after implementing working DIoU loss logic in the tensorflow backend, the CNN ultimately converged to a different value. The behaviour being shown suggested that the model was aiming to create the largest circle possible that contained the ground truth circle, with no regard given to the fact that this was leading to extremely small IoU values. After revisiting my logic and ensuring it was accurate, I spent some time trying to manually adjust the weights of the IoU and distance factors in the loss logic, but ultimately wasn't able to get the CNN to exhibit the behaviour I was after, so I decided to switch to Torch for some more control and granularity in training and fitting.