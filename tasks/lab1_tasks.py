import torch.optim as optim

from base import task, task_list
from models import *

lab1_task_list = task_list('/users/local/test')

lab1_task_list.AddTask(task({
                            'name': 'VGG11_lr0.01',
                            'model': lambda:VGG('VGG11'),
                            'optimizer': lambda net:optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4),
                            'useSubset': True
                           }))
                           
lab1_task_list.AddTask(task({
                            'name': 'VGG13_lr0.1',
                            'model': lambda:VGG('VGG13'),
                            'optimizer': lambda net:optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4),
                            'useSubset': True
                           }))

# lab1_task_list.AddTask(task({
#                             'name': 'VGG13_lr0.05',
#                             'model': lambda:VGG('VGG13'),
#                             'optimizer': lambda net:optim.SGD(net.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4),
#                             'useSubset': True
#                            }))

# lab1_task_list.AddTask(task({
#                             'name': 'VGG11_modified',
#                             'model': lambda:VGG('VGG11'),
#                             'useSubset': True
#                            }))

# lab1_task_list.AddTask(task({
#                             'name': 'VGG13_modified',
#                             'model': lambda:VGG('VGG13'),
#                             'useSubset': True
#                            }))

# lab1_task_list.AddTask(task({
#                             'name': 'VGG16_modified',
#                             'model': lambda:VGG('VGG16'),
#                             'useSubset': True
#                            }))
                           
# lab1_task_list.AddTask(task({
#                             'name': 'VGG19_modified',
#                             'model': lambda:VGG('VGG19'),
#                             'useSubset': True
#                            }))

# lab1_task_list.AddTask(task({
#                             'name': 'ResNet18',
#                             'model': lambda:ResNet18(),
#                             'useSubset': True
#                            }))