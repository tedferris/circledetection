This directory contains all the datasets and dataset-related logic needed for the CNN circle detection project.

**dataset_generation.py** - This script contains the logic that generates our train, test and validation datasets. It takes in two parameters:
- max_noise: Just as the name suggests, this param adjusts the max noise in our datasets. By default, it's set to 0.75. Each sample is generated with a random amount of noise between 0 and max_noise.
- num_images: The total number of samples to be generated. By default, num_samples is 2000. Keep in mind that this number represents the sum of train, test and validation, at a 60/20/20 split. For example, if num_samples is 5000, then there will be 3000 samples in our training set, and 1000 in each of the test and validation sets.

This script saves each sample input as its own .npy under the corresponding folder in data/datasets/, and  generates a .csv file for each of the train, test and validation sets containing ground truths. This .csv has four columns - the first, 'PATH', contains the path to that sample's .npy file. The next three, 'ROW', 'COL' and 'RAD', contain the circle parameters for that sample.