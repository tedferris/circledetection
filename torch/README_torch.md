This directory contains all the progress made towards training a Circle Detection CNN in Torch. The structure is slightly different than the TensorFlow approach.

**model.py** - This contains a custom CNN Class, built on torch.nn. This CNN architecture is slightly more complex than the TensorFlow approach, using two more convolutional layers and one more relu/linearization combo in the fully connected layer for a more gradual descent from high-dimensional space to 1x3 output.

**helpers.py** - Contains the given helper functions, as well as a Torch backend implementation of DIoU such that it's fully differentiable and yields a gradient. This can be used as a loss function in training, but similar to TensorFLow, was exhibiting some strange behaviour.

**train.py** - This script creates a CNN from the CNN Class, loads in our training set, sets up a DataLoader, and trains our model. All our hyperparameters can be tuned here, as well as training specs such as batch size, loss metric and number of epochs. The train() function in this script is also set up to print loss each epoch, as well as save a model checkpoint every 25 epochs.

**retrain.py** - Loads a CNN from a saved checkpoint and continues training. Useful for swapping metrics midway though a training run, or optimizing hyperparameters from early- to late-stage training.

**validation.py** - This script creates a new CNN, loads in a saved checkpoint, and runs a validation check on that model using the given IoU function. It again uses a DataLoader for simplicity, and returns the mean IoU for each validation run (using the same batch size as the training runs).
