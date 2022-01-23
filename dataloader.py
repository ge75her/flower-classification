import torch
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
import os
import cv2
import numpy as np

# load data
def read_file(path):
    image_dir=sorted(os.listdir(path))
    x = np.zeros((len(image_dir)-1, 256,256, 3),dtype = np.uint8)
    y = np.zeros(len(image_dir)-1,dtype = np.uint8)
    for i, file in enumerate(image_dir[1:]):
        img=cv2.imread(os.path.join(path,file))
        x[i,:,:]=cv2.resize(img,(256,256))
        if 'cat' in file.split('.'):
            y[i]=0
        else:
            y[i]=1
    return x,y


X_tcat,Y_tcat=read_file('flowers/flower_photos/train')
X_tdog,Y_tdog=read_file('training_set/training_set/dogs')
X_vcat,Y_vcat=read_file('test_set/test_set/cats')
X_vdog,Y_vdog=read_file('test_set/test_set/dogs')

X_train=np.concatenate((X_tcat,X_tdog),axis=0)
y_train=np.concatenate((Y_tcat,Y_tdog),axis=0)
X_val=np.concatenate((X_tcat,X_tdog),axis=0)
y_val=np.concatenate((Y_tcat,Y_tdog),axis=0)

# split train/val set
#from sklearn.model_selection import train_test_split
#X_train,X_test,y_train,y_test=train_test_split(x_train,y_train,train_size=0.7)
#print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)

#dataset
train_transform=transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.2,0.2,0.2))
])
test_transform=transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.2,0.2,0.2))
])


class Imgdataset(Dataset):
    def __init__(self,x,y,transforms):
        self.x=x
        self.y=torch.LongTensor(y)
        self.transforms=transforms
    def __len__(self):
        return len(self.x)
    def __getitem__(self, item):
        x=self.x[item]
        x=self.transforms(x)
        y=self.y[item]
        return x,y


train_set=Imgdataset(X_train,y_train,train_transform)
val_set=Imgdataset(X_val,y_val,test_transform)
train_loader=DataLoader(train_set,batch_size=32,shuffle=True)
val_loader=DataLoader(val_set,batch_size=32,shuffle=False)