import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import datasets,transforms
import os


data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "val": transforms.Compose([transforms.Resize((224, 224)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}


image_path = os.path.join('flowers','flower_photos')# flower data set path
#data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
#image_path = os.path.join(data_root, "data_set", "flower_data")  # flower data set path
train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),transform=data_transform["train"])
validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "validation"),transform=data_transform["val"])
train_num = len(train_dataset)
val_num = len(validate_dataset)
flower_list = train_dataset.class_to_idx
print(flower_list)


train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=32,shuffle=True)
validate_loader = torch.utils.data.DataLoader(validate_dataset,batch_size=32, shuffle=False)
print("using {} images for training, {} images for validation.".format(train_num,val_num))

test_data_iter = iter(validate_loader)
test_image, test_label = test_data_iter.next()
print(test_image.shape,test_label.shape)


