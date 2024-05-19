import os
import numpy as np
import torch
import pandas as pd
from PIL import Image
from IPython.display import display
import json

class TowerDataset2(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.transforms = transforms
        self.dataset = pd.read_csv(root)

    def __getitem__(self, idx):
        
        img = Image.open(self.dataset.iloc[idx]['image_path'])
        objects_parsed = json.loads(self.dataset.iloc[idx]['objects'])
        imagename = self.dataset.iloc[idx]['image']
                
        # Get the boundingboxes for the selected image 
        boxes=[]
        labels=[]
        
        for bbox_dict in objects_parsed:
            boxes.append(bbox_dict['box'])
            labels.append(bbox_dict['class'])
        
        # Convert the labels first to an int 
        labels = list(map(int, labels))
            
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # Set iscrowd to zero for each boundingbox
        iscrowd = torch.zeros((labels.shape[0],), dtype=torch.int64)
        
        # Define the area of the bounding boxes 
        if (len(boxes)>0):
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])            
        else:
            area = torch.zeros((0,4), dtype=torch.float32)
        
        # Define the target 
        target = {} 
        target["boxes"] = boxes 
        target["image_id"] = torch.tensor([idx])
        target["labels"] = labels
        target["iscrowd"] = iscrowd
        target["area"] = area
        
        # To be clarified? 
        if self.transforms is not None:  
            img = self.transforms(img)
        
        # Return the results as tuple 
        return img, target
                 
    
    def __len__(self):
        return (self.dataset.shape[0])
    
    def getitem_name(self, imagename): 
        #Search index 
        filtered_df = self.dataset[self.dataset['image'] == imagename]
        return self.__getitem__(filtered_df.index[0])