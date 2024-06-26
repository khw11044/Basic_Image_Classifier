import os
import glob
import csv
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from transform import create_train_transform, create_validation_transform
import torch

class MyDataset(Dataset):
    def __init__(self, root='./dataset/train', transform = None, mode='train', classes=None): # mode: train or vaild
        super(MyDataset, self).__init__()
        self.root = root
        # data_path = glob.glob(root+'/**/*.jpg')
        data_path = []
        for cls in classes:
            cls_img_path = glob.glob(os.path.join(root,cls) + '/*')
            DATA_LEN = int(len(cls_img_path)*0.9)
            if mode=='train':
                data_path += cls_img_path[:DATA_LEN+1]
            else:
                data_path += cls_img_path[DATA_LEN:]
        
        self.transform = transform
        self.mode = mode
        if classes is None:
            raise Exception('needed classes')
        self.labels = []
        self.images = []
        self.index_labels = {cls:0 for cls in classes}
        for i, cls in enumerate(classes):
            self.index_labels[cls]=i
        

        for line in data_path:
            image_path = line
            label = line.split('/')[-2]
            self.images.append(image_path)
            self.labels.append(label)
    def __getitem__(self, index):
        img, label = self.images[index], self.labels[index]
        img = self.read_image(img)
        label = self.convert_label(label)
        if self.transform:
            img = self.transform(image = img)['image']
            return img, label
    def __len__(self):
        return len(self.images)

    def read_image(self, img):
        img = cv2.imread(img, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    def convert_label(self, label):
        return self.index_labels[label]

    def get_class_weights(self):
        weights = [0 for _ in range(len(self.index_labels))]
        for label in self.labels:
            weights[self.index_labels[label]]+=1
        weights = np.array(weights)
        weights = weights / weights.sum()
        weights = 1.0 / weights
        weights = weights / weights.sum()
        return weights

    def get_class_weights2(self):
        weights = [0 for _ in range(len(self.index_labels))]
        for label in self.labels:
            weights[self.index_labels[label]]+=1
        weights = np.array(weights)
        normedWeights = [1 - (x / sum(weights)) for x in weights]
        return normedWeights

def mean_std(train_dataloader):
    mean0 = 0
    mean1 = 0
    mean2 = 0
    std0 = 0
    std1 = 0
    std2 = 0

    for image, _ in train_dataloader:
        mean0+=image[:,0,:,:].mean()
        mean1+=image[:,1,:,:].mean()
        mean2+=image[:,2,:,:].mean()
        std0+=image[:,0,:,:].std()
        std1+=image[:,1,:,:].std()
        std2+=image[:,2,:,:].std()

    print(mean0/len(train_dataloader))
    print(mean1/len(train_dataloader))
    print(mean2/len(train_dataloader))
    print(std0/len(train_dataloader))
    print(std1/len(train_dataloader))
    print(std2/len(train_dataloader))
    
    
if __name__=='__main__':

    classes = set()

    train_path = '../train'
    test_path = '../test'

    total_train_num = 0
    total_test_num = 0
    for label in os.listdir(train_path):
        classes.add(label)
        image_num = len(os.listdir(os.path.join(train_path,label)))
        total_train_num += image_num
        print('train dataset size : {} -> {}'.format(label,image_num))

    train_transform = create_train_transform(True, True, True, True)
    train_dataset = MyDataset(transform = train_transform, classes = classes)
    train_dataloader = DataLoader(train_dataset, batch_size=16)
    mean_std(train_dataloader)
    '''
    classes = set()

    train_path = './train'
    test_path = './test'

    total_train_num = 0
    total_test_num = 0
    for label in os.listdir(train_path):
        classes.add(label)
        image_num = len(os.listdir(os.path.join(train_path,label)))
        total_train_num += image_num
        print('train dataset size : {} -> {}'.format(label,image_num))

    for label in os.listdir(test_path):
        image_num = len(os.listdir(os.path.join(test_path,label)))
        total_test_num += image_num
        print('test dataset size : {} -> {}'.format(label,image_num))
        print('total train dataset : {} \t total test dataset : {}'.format(total_train_num, total_test_num))      


    train_transform = create_train_transform(True, True, True, True)
    train_dataset = MyDataset(transform = train_transform, classes = classes)
    val_transform = create_validation_transform(True)
    val_dataset = MyDataset(transform = val_transform, mode = 'validation', classes = classes)
    test_dataset = MyDataset(transform = val_transform, mode = 'test', classes = classes)
    print(len(train_dataset))
    print(len(val_dataset))
    print(len(test_dataset))
    print(train_dataset.get_class_weights())
    print(train_dataset.get_class_weights2())
    '''
