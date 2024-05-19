import os
import numpy as np
import torch
import pandas as pd
from PIL import Image
from IPython.display import display

class TowerDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.transforms = transforms
        self.dataset = pd.read_csv(root)
        self.dataset_unique = self.dataset.drop_duplicates(subset=['pylone', 'filename', 'filepath'])
            
    def __getitem__(self, idx):
        
        img = Image.open(self.dataset_unique.iloc[idx]['filepath'])

        # Get the boundingboxes for the selected image 
        boxes=[]
        columns_to_select = ['topleftx', 'toplefty', 'bottomrightx', 'bottomrighty']
        filtered_df = self.dataset[self.dataset['filename'] == self.dataset_unique.iloc[idx]['filename']]
        for index, row in filtered_df.iterrows():
            xmin = row['bottomrightx']
            xmax = row['topleftx']
            ymin = row['bottomrighty']
            ymax = row['toplefty']
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
        # Get the labels for each boundingbox 
        labels_column_to_convert = ['nature_int']
        labels = [list(row) for row in filtered_df[labels_column_to_convert].values]
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # Set iscrowd to zero for each boundingbox
        iscrowd = torch.zeros((labels.shape[0],), dtype=torch.int64)
        
        # Define the area of the bounding boxes 
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])            
        
        # Define the target 
        target = {} 
        target["boxes"] = boxes 
        target["image_id"] = self.dataset_unique.iloc[idx]['filename']
        target["labels"] = labels
        target["crowd"] = iscrowd
        target["area"] = area
        
        # To be clarified? 
        if self.transforms is not None:  
            img, target = self.transforms(img, target)
        
        # Return the results as tuple 
        return img, target
                 
    
    def __len__(self):
        return (self.dataset_unique.shape[0])
    
    def getitem_name(self, filename): 
        #Search index 
        filtered_df = self.dataset_unique[self.dataset_unique['filename'] == filename]
        return self.__getitem__(filtered_df.index[0])