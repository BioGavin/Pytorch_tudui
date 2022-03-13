import torch
import torchvision

# data preparation
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *

train_data = torchvision.datasets.CIFAR10(root="../dataset", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="../dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)
print(f'train_data_size: {train_data_size}')
print(f'test_data_size: {test_data_size}')

# data loader
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# build model
tudui = Tudui()
if torch.cuda.is_available():
    tudui = tudui.cuda()

# loss function
loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()

# optimizer
learning_rate = 1e-2
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate)

# parameter
total_train_step = 0
total_test_step = 0
epoch = 5

# tensorboard
writer = SummaryWriter('../train_log')


for i in range(epoch):
    print(f'-----epoch {i + 1} start-----')
    for data in train_dataloader:
        imgs, targets = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()
        outputs = tudui(imgs)
        loss = loss_fn(outputs, targets)
        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step += 1
        if total_train_step % 100 == 0:
            print(f'total_train_step: {total_train_step}, Loss: {loss.item()}')
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # test process
    total_test_loss = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            outputs = tudui(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
    print(f'total_test_loss: {total_test_loss}')
    total_test_step += 1
    writer.add_scalar("test_loss", total_test_loss, total_test_step)

    torch.save(tudui, "tudui_{}.pth".format(i))
    print("model saved")

writer.close()