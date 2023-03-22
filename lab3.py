import torch
import torch.optim as optim
import torch.nn.utils.prune as prune

import base
from models import *

device = base.CheckEnvironment()

################################### Part 1 ###################################

# We load the dictionnary
loaded_cpt = torch.load("/users/local/lab1/test2/checkpoint/VGG16.pth")

# Define the model 
model = VGG('VGG16')

model = torch.nn.DataParallel(model)

# Finally we can load the state_dict in order to load the trained parameters 
model.load_state_dict(loaded_cpt['net'])

trainloader, trainloader_subset, testloader = base.PrepareData()

criterion = nn.CrossEntropyLoss()

# Verifify previous accuracy
base.ManuelTest(device, model, criterion, testloader)

# parameters_to_prune = []
# for m in model.modules():
#             if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
#                 parameters_to_prune.append((m, 'weight'))

# prune.global_unstructured(tuple(parameters_to_prune), pruning_method=prune.L1Unstructured,amount=0.95)

# for m in model.modules():
#             if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
#                 print("Sparsity : {:.2f}%".format(100. * float(torch.sum(m.weight == 0))/ float(m.weight.nelement())))

# base.ManuelTest(device, model, criterion, testloader)

# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
# for i in range(0,10):
#     base.ManuelTrain(device, model, optimizer, criterion, trainloader_subset)
#     base.ManuelTest(device, model, criterion, testloader)