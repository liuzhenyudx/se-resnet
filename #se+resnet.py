#se+resnet
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import optim as optim
import visdom
import numpy as np
class basicblock(nn.Module):
	"""docstring for basicblock"""
	def __init__(self, inchannel,outchannel,stride=1):
		super(basicblock, self).__init__()
		self.main=nn.Sequential(
			nn.Conv2d(inchannel,outchannel,3,stride=stride,padding=1,bias=False),
			nn.BatchNorm2d(outchannel),
			nn.ReLU(inplace=True),
			nn.Conv2d(outchannel,outchannel,3,stride=1,padding=1,bias=False),
			nn.BatchNorm2d(outchannel)
			)
		self.shortcut=nn.Sequential()
		if stride!=1 or inchannel!=outchannel :
			self.shortcut=nn.Sequential(
				nn.Conv2d(inchannel,outchannel,1,stride=stride,bias=False),
				nn.BatchNorm2d(outchannel))
		if outchannel == 64:
			self.pool=nn.AvgPool2d(32,stride=1)
			
		elif outchannel ==128:
			self.pool=nn.AvgPool2d(16,stride=1)
			
		elif outchannel == 256:
			self.pool=nn.AvgPool2d(8,stride=1)
		elif outchannel == 512 :
			self.pool=nn.AvgPool2d(4,stride=1)
		self.fc1=nn.Sequential(nn.Dropout(),
            nn.Linear(outchannel,outchannel//16))
		self.relu=nn.ReLU()
		self.fc2=nn.Sequential(nn.Dropout(),
            torch.nn.Linear(outchannel//16,outchannel))
		self.sigmoid=nn.Sigmoid()

	def forward(self,input):
		output=self.main(input)
		short=self.shortcut(input)
		output1=self.pool(output)
		output1=output1.view(output1.size(0),-1)
		output1=self.fc1(output1)
		output1=self.relu(output1)
		output1=self.fc2(output1)
		output1=self.sigmoid(output1)
		output1=output1.view(output1.size(0),output1.size(1),1,1)
		out=output*output1
		out=short+out
		out=nn.functional.relu(out)
		return out

class resnet(nn.Module):
	"""docstring for resnet"""
	def __init__(self, arg):
		super(resnet, self).__init__()
		self.inchannel=64
		self.conv=nn.Sequential(
			nn.Conv2d(3,64,3,1,1,bias=False),
			nn.BatchNorm2d(self.inchannel),
			nn.ReLU(inplace=True))
		self.layer1=self.make(basicblock,64,2,stride=1)
		self.layer2=self.make(basicblock,128,2,stride=2)
		self.layer3=self.make(basicblock,256,2,stride=2)
		self.layer4=self.make(basicblock,512,2,stride=2)
		self.fc = nn.Sequential(nn.Dropout(),
            nn.Linear(512,100))
	def make(self,block,channel,num,stride):
		strides=[stride] + [1]*(num-1)
		layers = []
		for stride in strides:
			layers.append(block(self.inchannel,channel,stride))
			self.inchannel=channel
		return nn.Sequential(*layers)
	def forward(self,input):
		output = self.conv(input)
		output = self.layer1(output)
		output = self.layer2(output)
		output = self.layer3(output)
		output = self.layer4(output)
		output = torch.nn.functional.avg_pool2d(output,4)
		output = output.view(output.size(0),-1)
		output = self.fc(output)
		return output

def network():
	return resnet(basicblock)
	pass
		



def train(model, data, target, loss_func, optimizer):
    """
    train step, input a batch data, return accuracy and loss
    :param model: network model object
    :param data: input data, shape: (batch_size, 28, 28, 1)
    :param target: input labels, shape: (batch_size, 1)
    :param loss_func: the loss function you use
    :param optimizer: the optimizer you use
    :return: accuracy, loss
    """
    model.train()
    # initial optimizer
    optimizer.zero_grad()

    # net work will do forward computation defined in net's [forward] function
    output = model(data)

    # get predictions from outputs, the highest score's index in a vector is predict class
    predictions = output.max(1, keepdim=True)[1]

    # cal correct predictions num
    correct = predictions.eq(target.view_as(predictions)).sum().item()

    # cal accuracy
    acc = correct / len(target)

    # use loss func to cal loss
    loss = loss_func(output, target)

    # backward will back propagate loss
    loss.backward()

    # this will update all weights use the loss we just back propagate
    optimizer.step()

    return acc, loss


def test(model, test_loader, loss_func, use_cuda):
    """
    use a test set to test model
    NOTE: test step will not change network weights, so we don't use backward and optimizer
    :param model: net object
    :param test_loader: type: torch.utils.data.Dataloader
    :param loss_func: loss function
    :return: accuracy, loss
    """
    model.eval()
    acc_all = 0
    loss_all = 0
    step = 0
    with torch.no_grad():
        for data, target in test_loader:
            step += 1
            if use_cuda:
                data = data.cuda()
                target = target.cuda()
            output = model(data)
            predictions = output.max(1, keepdim=True)[1]
            correct = predictions.eq(target.view_as(predictions)).sum().item()
            acc = correct / len(target)
            loss = loss_func(output, target)
            acc_all += acc
            loss_all += loss
    return acc_all / step, loss_all / step


def main():
    """
    main function
    """

    # define some hyper parameters
    num_classes = 100
    eval_step = 1000
    num_epochs = 100
    batch_size = 64
    model_name = 'vgg' # resnet, vgg

    # first check directories, if not exist, create
    dir_list = ('../data', '../data/MNIST', '../data/CIFAR-100')
    for directory in dir_list:
        if not os.path.exists(directory):
            os.mkdir(directory)

    # if cuda available -> use cuda
    use_cuda = torch.cuda.is_available()
    # this step will create train_loader and  test_loader

    train_loader = DataLoader(
        datasets.CIFAR100(root='../liu', train=True, download=True,transform=
                         transforms.Compose([
                                            transforms.RandomCrop(32, padding=4),  #先四周填充0，在吧图像随机裁剪成32*32
                                            transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5017, 0.4867,0.4408), (0.2675,0.2565,0.2761))])),
                                            batch_size=batch_size,
                                            shuffle=True
                                            )

    test_loader = DataLoader(
        datasets.CIFAR100(root='../liu', train=False,
                         transform=transforms.Compose([transforms.ToTensor(),
                         transforms.Normalize((0.5017, 0.4867,0.4408), (0.2675,0.2565,0.2761))])),
        batch_size=batch_size
    )

    # define network
    model = network()
    if use_cuda:
        model = model.cuda()
    print(model)
    # define loss function
    ce_loss = torch.nn.CrossEntropyLoss()
    vis=visdom.Visdom()
    x,train_acc,test_acc=0,0,0
    win=vis.line(
        X=np.array([x]),
        Y = np.column_stack((np.array([train_acc]),np.array([test_acc]))),
        opts=dict(
            title = "train ACC and test ACC",
            legend =["train_acc","test_acc"])
        )


    # define optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # start train
    train_step = 0
    for _ in range(num_epochs):
        for data, target in train_loader:
            train_step += 1
            if use_cuda:
                data = data.cuda()
                target = target.cuda()
            acc, loss = train(model, data, target, ce_loss, optimizer)
            train_acc=acc
            if train_step % 5 == 0:
                print('Train set: Step: {}, Loss: {:.4f}, Accuracy: {:.2f}'.format(train_step, loss, acc))
                acc, loss = test(model, test_loader, ce_loss, use_cuda)
                print('\nTest set: Step: {}, Loss: {:.4f}, Accuracy: {:.2f}\n'.format(train_step, loss, acc))
                test_acc=acc
                vis.line(
                    X = np.array([train_step]),
                    Y = np.column_stack(
                        (np.array([train_acc]),np.array([test_acc])
                        )
                        ),
                    win = win,
                    update = "append"
                    )
                pass


if __name__ == '__main__':
    main()
