import os
import cv2
import pandas as pd

def imgtopix(directory):
    class_ = os.listdir(directory)
    pixels = []
    labels = []
    for label in class_:
        images = os.listdir(directory+label+'/')
        for image in images:
          img = cv2.imread(directory+label+'/'+image)
          img = cv2.resize(img, (224, 224))
          img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
          pixels.append(torch.tensor(img.tolist()))
          labels.append(labels)
    dataframe = pd.DataFrame({'pixels': pixels, 'labels': labels})      
    return dataframe

LOCATION = 'data/train-data/'

data = imgtopix(LOCATION)
data.to_csv('pixelvslabels.csv')
