import torch
import numpy as np
import pandas as pd
import rasterio
import os


class CustomDataGenerator(torch.utils.data.Dataset):
    def __init__(self, path, transform=None):
        # read file names from csv files
        csv_path = '{}/train_meta.csv'.format(path)
        df = pd.read_csv(csv_path)    
        img, mask = df.columns.tolist()
        
        TRAIN_IMAGE = path + '/train_img'
        TRAIN_MASK =  path + '/train_mask'
        
        self.images = [os.path.join(TRAIN_IMAGE, x) for x in df[img].tolist()]
        self.masks = [os.path.join(TRAIN_MASK, x) for x in df[mask].tolist()]

        self.MAX_PIXEL_VALUE = 65535                                            # defined in the original code
        self.transform = transform                                              # image transforms for data augmentation

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img = rasterio.open(self.images[idx]).read(chanels_num)               # only extract 3 channels  
        img = np.float32(img.transpose((1, 2, 0))) / self.MAX_PIXEL_VALUE

        mask = rasterio.open(self.masks[idx]).read().transpose((1, 2, 0))
        sample = {'image': img, 'mask': mask}
        
        if self.transform:
            sample = self.transform(sample)

        return sample