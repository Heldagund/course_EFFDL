from base import GenTaskList
from models import *
import torch.optim as optim

# def GetTrainedModel(modelBuilder):
#   net = modelBuilder()

#   netData = torch.load('/users/local/lab2/pretrained_model/checkpoint/DenseNet169.pth')
#   net.load_state_dict(netData['net'])

#   return net

config = [
          {'param': 'model', 
           'options': [
                        # {'tag': 'ResNet18',
                        #  'value': lambda:ResNet18(),
                        # },
                        # {'tag': 'ResNet34',
                        #  'value': lambda:ResNet34(),
                        # },
                        # {'tag': 'ResNet50',
                        #  'value': lambda:ResNet50()
                        # },
                        # {'tag': 'DenseNet121',
                        #  'value': lambda:DenseNet121()
                        # },
                        # {'tag': 'DenseNet169',
                        #  'value': lambda:DenseNet169()
                        # }
                        # {'tag': 'DenseNet169_Retrain',
                        #  'value': lambda: GetTrainedModel(lambda:DenseNet169())
                        # }
                      ]
          },
          {'param': 'optimizer',
           'options': [
                        {
                          'tag': 'lr0.001',
                          'value': lambda net:optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
                        }
                      ] 
          }
         ]

pretrain_task_list = GenTaskList(config, '/users/local/lab2/pretrained_model')