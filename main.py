import torch
from dataset import MyDataset
from model import MyModel
from loss import MyLoss1, MyLoss2
from transform import create_train_transform, create_validation_transform
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from activate import train, validation

device = torch.device('cuda:0')


save_path = '../result'
os.makedirs(save_path, exist_ok=True)
save_model = save_path + '/best_model.pth'

BATCHSIZE = 64
LR = 0.01
EPOCHS = 20
train_path = '../dataset/train'
test_path = '../dataset/test'


def get_classes(train_path, test_path):
    classes = set()

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
    return classes

classes = get_classes(train_path, test_path)
num_classes = len(classes)

train_transform = create_train_transform(True, True, True, True)
train_dataset = MyDataset(root=train_path, transform = train_transform, mode = 'train', classes = classes)
validation_transform = create_validation_transform(True)
validation_dataset = MyDataset(root=train_path, transform = validation_transform, mode = 'validation', classes = classes)


model = MyModel(num_classes = len(classes)).to(device)
criterion = MyLoss1(weights = torch.tensor(train_dataset.get_class_weights(), dtype=torch.float32).to(device))
#criterion = MyLoss2(weights = torch.tensor(train_dataset.get_class_weights2(), dtype=torch.float32).to(device))
optimizer = optim.SGD(model.parameters(), lr = LR)
#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor = 0.5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
#scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,15,20,25,30], gamma=0.1)

train_dataloader = DataLoader(train_dataset, batch_size = BATCHSIZE,shuffle = True, num_workers=4)
validation_dataloader = DataLoader(validation_dataset, batch_size = BATCHSIZE,shuffle = False)

pre_score = 0

for epoch in range(EPOCHS):
    # 훈련 진행 
    train(model, train_dataloader, criterion, optimizer, device)
    
    # 5 epoch마다 vaildation 진행 
    if (epoch+1)%5==0:
        score = validation(model, validation_dataloader, criterion,  None, device)
        if score>pre_score:
            pre_score = score
            model = model.cpu()
            print('---'*10)
            print('best score: {} and save model'.format(score))
            torch.save(model.state_dict(), save_model)
            model = model.to(device)
        scheduler.step(score)


