###################### Part 1 ######################
# from torchvision.datasets import CIFAR10
# from torch.utils.data.dataloader import DataLoader

# import torchvision.transforms as transforms

# transform_train = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor()
# ])

# rootdir = '/users/local'

# c10train = CIFAR10(rootdir,train=True,download=True,transform=transform_train)

# trainloader = DataLoader(c10train,batch_size=4,shuffle=False) ### Shuffle to False so that we always see the same images

# from matplotlib import pyplot as plt 

# ### Let's do a figure for each batch
# f = plt.figure(figsize=(10,10))

# for i,(data,target) in enumerate(trainloader):
    
#     data = (data.numpy())
#     print(data.shape)
#     plt.subplot(2,2,1)
#     plt.imshow(data[0].swapaxes(0,2).swapaxes(0,1))
#     plt.subplot(2,2,2)
#     plt.imshow(data[1].swapaxes(0,2).swapaxes(0,1))
#     plt.subplot(2,2,3)
#     plt.imshow(data[2].swapaxes(0,2).swapaxes(0,1))
#     plt.subplot(2,2,4)
#     plt.imshow(data[3].swapaxes(0,2).swapaxes(0,1))

#     break

# f.savefig('train_DA.png')

import base
from tasks import lab4_task_list

def main():
    isResume = base.ParseCommand()
    trainloader, trainloader_subset, testloader = base.PrepareData()
    device = base.CheckEnvironment()
    base.RunTasks(isResume, device, trainloader, trainloader_subset, testloader, lab4_task_list)
    # base.PlotResult(lab4_task_list)
    # base.SendResult(lab4_task_list)

if __name__ == '__main__':
    main()