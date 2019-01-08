# Script to seperate dataset according to requirements of PyTorch ImageLoader
# root_directory/classx/image.jpg

import os
import errno
import shutil
import pandas as pd

IMAGES = 'data/train-data/train/'
DIR = 'data/train-data/'

# Classes to be predicted
# buildings-0,forest-1,glacier-2,mountain-3,sea-4,street-5
classes = ['0', '1', '2', '3', '4', '5']

# Make directories
for class_ in classes:
    if not os.path.exists(DIR+class_):
        try:
            os.makedirs(DIR+class_)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    
# Import training labels
data = pd.read_csv(DIR+'train.csv')

# Moving files in respective directories
for images, labels in zip(data['image_name'], data['label']):
    shutil.move(IMAGES+images, DIR+str(labels)+'/'+images)
