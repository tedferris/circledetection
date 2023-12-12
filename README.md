This README serves as a general overview of the CNN circle detection repo.

There are three sub-directories in the repo, each with their own README:
- The data directory contains dataset_generation.py, a generation script that creates training, test and validation data in a 60/20/20 split.
- The tensorflow directory contains my first attempt at training a CNN in TensorFlow on Keras. I wasn't able to find a huge amount of success with TensorFlow, so I pivoted to Torch for the added granularity and functionality.
- The torch directory contains the most recent progress in model training. 
