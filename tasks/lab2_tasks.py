import torch.optim as optim

from base import task, task_list
from models import *
from methods import BC, GenBCModel, BCTrainRoutine, BCTestRoutine

lab2_task_list = task_list('/users/local/lab2/test2')

lab2_task_list.AddTask(task({
                            'name': 'VGG13_BC_lr0.01',
                            'model': lambda: GenBCModel(lambda:VGG('VGG13')),
                            'optimizer': lambda net:optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4),
                            'train_routine': BCTrainRoutine,
                            'test_routine': BCTestRoutine,
                            'useSubset': False
                           }))

lab2_task_list.AddTask(task({
                            'name': 'ResNet18_BC_lr0.01',
                            'model': lambda: GenBCModel(lambda:ResNet18()),
                            'optimizer': lambda net:optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4),
                            'train_routine': BCTrainRoutine,
                            'test_routine': BCTestRoutine,
                            'useSubset': False
                           }))

lab2_task_list.AddTask(task({
                            'name': 'ResNet34_BC_lr0.01',
                            'model': lambda: GenBCModel(lambda:ResNet34()),
                            'optimizer': lambda net:optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4),
                            'train_routine': BCTrainRoutine,
                            'test_routine': BCTestRoutine,
                            'useSubset': False
                           }))

lab2_task_list.AddTask(task({
                            'name': 'DenseNet121_BC_lr0.01',
                            'model': lambda: GenBCModel(lambda:DenseNet121()),
                            'optimizer': lambda net:optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4),
                            'train_routine': BCTrainRoutine,
                            'test_routine': BCTestRoutine,
                            'useSubset': False
                           }))

# lab2_task_list.AddTask(task({
#                             'name': 'PreActResNet50_BC_Adam_lr0.001',
#                             'model': lambda: GenBCModel(lambda:PreActResNet50()),
#                             'optimizer': lambda net:optim.Adam(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4),
#                             'train_routine': BCTrainRoutine,
#                             'test_routine': BCTestRoutine,
#                             'useSubset': True
#                            }))

