from VGG import vgg
from data import train_loader,validate_loader,validate_dataset
import torch
import torch.nn as nn
from tqdm import tqdm
import sys
#del=torchvision.models.resnet50(pretrained=True)
#for param in model.parameters():
#   param.requires_grad=False
#num_ftrs=model.fc.in_features
#model.fc=nn.Linear(num_ftrs,2)
#model.to(device)
device=torch.device('cpu')
model_name = "vgg16"
net = vgg(model_name=model_name, num_classes=5, init_weights=True)
net.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

epochs = 30
best_acc = 0.0
save_path = './{}Net.pth'.format(model_name)
train_steps = len(train_loader)
val_num=len(validate_dataset)
for epoch in range(epochs):
        # train
    net.train()
    running_loss = 0.0
    train_bar = tqdm(train_loader, file=sys.stdout)
    for step, data in enumerate(train_bar):
        images, labels = data
        #print(labels)
        optimizer.zero_grad()
        outputs = net(images.to(device))
        #print(outputs)
        loss = loss_function(outputs, labels.to(device))
        loss.backward()
        optimizer.step()

            # print statistics
        running_loss += loss.item()

        train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # validate
    net.eval()
    acc = 0.0  # accumulate accurate number / epoch
    with torch.no_grad():
        val_bar = tqdm(validate_loader, file=sys.stdout)
        for val_data in val_bar:
            val_images, val_labels = val_data
            #print(val_labels)
            outputs = net(val_images.to(device))
            #print(outputs)
            predict_y = torch.max(outputs, dim=1)[1]
            acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

    val_accurate = acc / val_num
    print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

    if val_accurate > best_acc:
        best_acc = val_accurate
        torch.save(net.state_dict(), save_path)

print('Finished Training')
