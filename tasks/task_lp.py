import torch
import torch.optim as optim

from base import task, task_list
from models import *
from methods import BC, GenBCModel, BCTrainRoutine, BCTestRoutine

lp_task_list = task_list('/users/local/long_project/test')
lp_task_list.AddTask(task({
                            'name': 'PNASNet_lr0.01',
                            'model': lambda:TestPNASNet(),
                            'optimizer': lambda net:optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4),
                           }))
lp_task_list.AddTask(task({
                            "name": "PNASNet_nc128_lr0.01",
                            "model": lambda:TestPNASNet(128),
                            "optimizer": lambda net:optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4),
                            }))

lp_task_list.AddTask(task({
                            "name": "PNASNet_nc128_BC_lr0.01",
                            "model": lambda: GenBCModel(lambda:TestPNASNet(128)),
                            "optimizer": lambda net:optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4),
                            'train_routine': BCTrainRoutine,
                            'test_routine': BCTestRoutine
                            }))

lp_task_list.AddTask(task({
                            "name": "DenseNetCifar_BC_lr0.01",
                            "model": lambda: GenBCModel(densenet_cifar),
                            "optimizer": lambda net:optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4),
                            "train_routine": BCTrainRoutine,
                            "test_routine": BCTestRoutine
                            }))

def GetTrainedPNASNet():
    task = lp_task_list.GetTasks()["PNASNet_nc128_BC_lr0.01"]
    model = task.GetModel()
    netData = torch.load(task.ckptPath)
    model.load_state_dict(netData['net'])
    return model

lp_task_list.AddTask(task({
                            "name": "PNASNet_nc128_BC_lr0.001_Retrain",
                            "model": lambda: GetTrainedPNASNet(),
                            "optimizer": lambda net:optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4),
                            "train_routine": BCTrainRoutine,
                            "test_routine": BCTestRoutine
                            }))

def GetPrunedNet(modelPath, dataPath):
    model = torch.load(modelPath)
    data = torch.load(dataPath)
    model.load_state_dict(data['net'])
    return model

from methods import Mixup
def MixupBCTrainRoutine(net, optimizer, criterion, inputs, labels):
    optimizer.zero_grad()

    net.binarization()

    lam, inputsMixed, labelsPerm = Mixup(inputs, labels)

    outputs = net(inputsMixed)    
    loss = lam*criterion(outputs, labels) + (1-lam)*criterion(outputs, labelsPerm)
    loss.backward()

    net.restore()

    optimizer.step()

    net.clip()
    
    return outputs, loss

lp_task_list.AddTask(task({
                            "name": "densenet_cifar_pruned3_BC_lr0.01",
                            "model": lambda: GenBCModel(lambda: GetPrunedNet('/users/local/long_project/iter_pruning/2/Model/DenseNetCifar_Iter_3.pth',
                                                              '/users/local/long_project/iter_pruning/2/checkpoint/DenseNetCifar_lr0.002_Iter3.pth')),
                            "optimizer": lambda net:optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4),
                            "train_routine": MixupBCTrainRoutine,
                            "test_routine": BCTestRoutine
                            }))

lp_task_list.AddTask(task({
                            "name": "densenet_cifar_pruned2_BC_lr0.01",
                            "model": lambda: GenBCModel(lambda: GetPrunedNet('/users/local/long_project/iter_pruning/2/Model/DenseNetCifar_Iter_2.pth',
                                                              '/users/local/long_project/iter_pruning/2/checkpoint/DenseNetCifar_lr0.002_Iter2.pth')),
                            "optimizer": lambda net:optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4),
                            "train_routine": MixupBCTrainRoutine,
                            "test_routine": BCTestRoutine
                            }))

lp_task_list.AddTask(task({
                            "name": "densenet_cifar_pruned3_untrained_BC_lr0.01",
                            "model": lambda: GenBCModel(lambda: torch.load('/users/local/long_project/iter_pruning/2/Model/DenseNetCifar_Iter_3.pth')),
                            "optimizer": lambda net:optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4),
                            "train_routine": MixupBCTrainRoutine,
                            "test_routine": BCTestRoutine
                            }))

def GetPrunedBCNet(modelPath, dataPath):
    model = GenBCModel(lambda: torch.load(modelPath))
    data = torch.load(dataPath)
    model.load_state_dict(data['net'])
    return model

lp_task_list.AddTask(task({
                            "name": "densenet_cifar_pruned3_BC_lr0.001_r",
                            "model": lambda: GetPrunedBCNet('/users/local/long_project/iter_pruning/2/Model/DenseNetCifar_Iter_3.pth',
                                                              '/users/local/long_project/test/checkpoint/densenet_cifar_pruned3_BC_lr0.01.pth'),
                            "optimizer": lambda net:optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4),
                            "train_routine": MixupBCTrainRoutine,
                            "test_routine": BCTestRoutine
                            }))

lp_task_list.AddTask(task({
                            "name": "densenet_cifar_pruned3_BC_lr0.01_NoMix",
                            "model": lambda: GetPrunedBCNet('/users/local/long_project/iter_pruning/2/Model/DenseNetCifar_Iter_3.pth',
                                                              '/users/local/long_project/test/checkpoint/densenet_cifar_pruned3_BC_lr0.01.pth'),
                            "optimizer": lambda net:optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4),
                            "scheduler": lambda optimizer:optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200),
                            "train_routine": BCTrainRoutine,
                            "test_routine": BCTestRoutine
                            }))
