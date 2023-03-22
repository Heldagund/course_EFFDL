# import torch
# import math
# from methods import FP



# model = torch.load('/users/local/long_project/iter_pruning/2/Model/DenseNetCifar_Iter_3.pth')
# for m in model.modules():
#     if(isinstance(m, torch.nn.Conv2d)):
#         m.groups = math.gcd(m.in_channels, m.out_channels)
#         print(m.groups)
#         if(m.groups>1):
#             nparams_toprune = int(m.in_channels - m.in_channels/m.groups)
#             channels_toprune = FP._get_channels_toprune(1, m.weight.data, 1, nparams_toprune)
#             FP._prune_conv(1, m, 1, channels_toprune)

# torch.save(model, './grouped.pth')

from base import dispatcher
from FSM import FSM
from methods import MixupTrainRoutine
from base import task_list, task

import torch
import torch.optim as optim
import os
import re
import time

while(True):
    usage = int(re.sub('\D', '', os.popen('nvidia-smi | sed -n 10p').read().split('|')[3]))
    if(usage > 20):
        print('Sleeping......')
        time.sleep(60)
    else:
        break

pg_task_list = task_list('./../EFFDL_Data')
pg_task_list.AddTask(task({
                            "name": "DenseNetCifar_lr0.01_pruned_grouped",
                            "model": lambda: torch.load('./grouped.pth'),
                            "optimizer": lambda net:optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4),
                            "scheduler": lambda optimizer:optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200),
                            'train_routine': MixupTrainRoutine
                            }))

if __name__ == "__main__":
    top = FSM()
    globalDispatcher = dispatcher(top, taskList=pg_task_list)
    globalDispatcher.run()

