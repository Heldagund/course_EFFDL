import torch.optim as optim

from base import task, task_list
from models import *
from methods import MixupTrainRoutine

lab4_task_list = task_list('/users/local/lab4')
lab4_task_list.AddTask(task({
                            'name': 'VGG13_lr0.01',
                            'model': lambda:VGG('VGG13'),
                            'optimizer': lambda net:optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4),
                            'train_routine': MixupTrainRoutine,
                            'useSubset': True
                           }))

lab4_task_list.AddTask(task({
                            'name': 'ResNet18_lr0.01_Mixup',
                            'model': lambda:ResNet18(),
                            'optimizer': lambda net:optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4),
                            'train_routine': MixupTrainRoutine
                           }))