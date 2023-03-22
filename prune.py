import torch
import numpy as np

import base
from methods import FP_DenseNet, GenBCModel
from models import *

# for name, m in model.named_children():
#     print('{}:\n{}'.format(name, m))
# for idx, m in enumerate(fp.target_modules):
#     print('{}: {}'.format(idx, m))
# for idx, m in enumerate(fp.GetPrunableConvs()):
#     print('{}: {}'.format(idx, m))

# Prune but test and manually
# model = densenet_cifar()
model = torch.load('./grouped.pth')
# model = GenBCModel(lambda: torch.load('/users/local/long_project/iter_pruning/2/Model/DenseNetCifar_Iter_3.pth'))
# data = torch.load('/users/local/long_project/test/checkpoint/densenet_cifar_pruned3_BC_lr0.001_r.pth')
# model.load_state_dict(data['net'])

# fp = FP_DenseNet(model)
# cnt = len(fp.GetPrunableConvs())
# prune_rate_list = np.linspace(0, 0.5, cnt).tolist()
# fp.SetPruneRate(prune_rate_list)
# fp.PruneAll()

# with open('pruned.txt', 'w') as f:
#     for name, m in model.named_children():
#         f.write('{}:\n{}'.format(name, m))

device = 'cuda'
model.to(device)
# model.binarization()
_, _, testloader = base.PrepareData()
criterion = nn.CrossEntropyLoss()
base.ManuelTest(device, model, criterion, testloader, base.DefaultTestRoutine)

model.to('cpu')

from profile import profile
ref_params = 5586981
ref_flops  = 834362880
sparsity=0.

flops, params = profile(model, (1,3,32,32))
flops, params = flops.item(), params.item()

score_flops = flops / ref_flops
score_params = (params / ref_params)*(1-sparsity)
score = score_flops + score_params
print("Flops: {}, Params: {}".format(flops,params))
print("Score flops: {} Score Params: {}".format(score_flops,score_params))
print("Final score: {}".format(score))

# from base import sub_state, dispatcher, task, task_list
# from FSM import FSM, launch_task, start_task, train, test, save_and_step, end_task, clean, update_task_list
# import torch.optim as optim
# import os

# class iterative_pruning(launch_task):
#     def __init__(self):
#         super(iterative_pruning, self).__init__()

#         # Substates
#         self.SetSubstate(startTask = start_task(),
#                          testBefore = test(),
#                          pruneInit = prune_init(),
#                          prunePart = prune_part(),
#                          train = train(),
#                          testInloop = test(),
#                          saveAndStep = save_and_step(), 
#                          endTask = end_task())

#     def BuildControlFlow(self):
#         self.startTask >= self.testBefore >= self.pruneInit >= self.prunePart >= self.train >= self.testInloop >= self.saveAndStep >= self.train
#         self.saveAndStep >= self.endTask
#         return self.startTask
    
#     def GetNextState(self, subDataflow):
#         if subDataflow['isKeyInterrupt']:
#             nextState = self.GetStateByClass(clean)
#         else:
#             nextState = self.GetStateByClass(update_task_list)
#         return nextState, subDataflow

#     def Action(self):       # Overload Action in order to deal with exception
#         entryState = self.BuildControlFlow()
#         localDispatcher = dispatcher(entryState, **self.dataflow)
#         try:
#             subDataflow = localDispatcher.run()
#         except KeyboardInterrupt:
#             subDataflow = dict(isKeyInterrupt=True)
#         nextState, finalDataflow = self.GetNextState(subDataflow)
#         self.AddToDataflow(**finalDataflow)
#         return nextState

# class prune_init(sub_state):
#     def Action(self):
#         self.__PruneInit()
#         return self.nextState[0]

#     def __PruneInit(self):
#         if not self.needResume:
#             fp = FP_DenseNet(self.net)
#             cnt = len(fp.GetPrunableConvs())
#             prune_rate_list = np.linspace(0.1, 0.6, cnt).tolist()
#             self.AddToDataflow(pruner=fp, cnt=cnt, prune_rate_list=prune_rate_list)

#         self.best_acc = 0


# class prune_part(sub_state):
#     def Action(self):
#         self.__PrunePart()
#         return self.nextState[0]

#     def __PrunePart(self):
#         if not self.needResume:
#             beginIdx = round(self.cnt*2/3)
#             endIdx = round(self.cnt)
#             self.net.to('cpu')
#             self.pruner.PrunePart(range(beginIdx, endIdx), self.prune_rate_list[beginIdx:endIdx])
#             if not os.path.isdir(self.taskList.root + '/Model'):
#                 os.mkdir(self.taskList.root + '/Model')
#             torch.save(self.net, self.taskList.root + '/Model/DenseNetCifar_Iter_3.pth')
#             with open('pruned.txt', 'w') as f:
#                 for name, m in self.net.named_children():
#                     f.write('{}:\n{}'.format(name, m))
#             self.net.to(self.device)

# class FSM_IP(FSM):
#     def __init__(self):
#         super(FSM_IP, self).__init__()
#         self.execution.SetSubstate(launchTask = iterative_pruning())
#         self.SetSubstate(analysis = None)

# def LoadNet():
#     model = torch.load('/users/local/long_project/iter_pruning/2/Model/DenseNetCifar_Iter_2.pth')
#     data = torch.load('/users/local/long_project/iter_pruning/2/checkpoint/DenseNetCifar_lr0.002_Iter2.pth')
#     model.load_state_dict(data['net'])
#     return model

# ip_task_list = task_list('/users/local/long_project/iter_pruning/2')
# ip_task_list.AddTask(task({
#                             "name": "DenseNetCifar_lr0.002_Iter3",
#                             "model": LoadNet,
#                             "optimizer": lambda net:optim.SGD(net.parameters(), lr=0.002, momentum=0.9, weight_decay=5e-4),
#                             "scheduler": lambda optimizer:optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)
#                             }))

# if __name__ == "__main__":
#     top = FSM_IP()
#     globalDispatcher = dispatcher(top, taskList=ip_task_list)
#     globalDispatcher.run()