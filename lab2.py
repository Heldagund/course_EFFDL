import torch

import base
from tasks import lab2_task_list
from models import *

device = base.CheckEnvironment()

################################### Part 1 ###################################
# from utils import progress_bar

# # We load the dictionnary
# loaded_cpt = torch.load("/users/local/lab1/test3/checkpoint/VGG11_modified.pth")

# # Define the model 
# model = VGG('VGG11')

# model = torch.nn.DataParallel(model)

# # Finally we can load the state_dict in order to load the trained parameters 
# model.load_state_dict(loaded_cpt['net'])

# trainloader, trainloader_subset, testloader = base.PrepareData()

# criterion = nn.CrossEntropyLoss()

# # Verifify previous accuracy
# base.ManuelTest(device, model, criterion, testloader)


# model.half()

# test_loss = 0
# correct = 0
# total = 0
# with torch.no_grad():
#     for batch_idx, (inputs, labels) in enumerate(testloader):
#         inputs = inputs.half()
#         inputs, labels = inputs.to(device), labels.to(device)
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)

#         test_loss += loss.item()
#         _, predicted = outputs.max(1)
#         total += labels.size(0)
#         correct += predicted.eq(labels).sum().item()

#         msg = 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total)
#         progress_bar(batch_idx, len(testloader), msg)

################################### Part 2 ###################################

# isResume = base.ParseCommand()
# trainloader, trainloader_subset, testloader = base.PrepareData()
# device = base.CheckEnvironment()
# base.RunTasks(isResume, device, trainloader, trainloader_subset, testloader, lab2_task_list)
# base.PlotResult(lab2_task_list)
# base.SendResult(lab2_task_list)

def SaveBCModel(model, ckptPath, savePath):
    model.binarization()
    layers = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            layer = {}
            mask = torch.ones_like(module.weight.data)
            layer['weight'] = torch.isclose(module.weight.data, mask)
            if not module.bias == None:
                layer['bias'] = module.bias.data.clone()
            layers[name] = layer
        else:
            if len(list(module.children()))==0:
                layers[name] = module.state_dict()
    torch.save(layers, savePath)

def RestoreBCModel(model, modelData):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            if name in modelData:
                layer = modelData[name]
                tmp = torch.sign(torch.add(layer['weight'], -0.1))
                module.weight.data.copy_(tmp)
                if 'bias' in layer:
                    module.bias.data.copy_(layer['bias'])
        else:
            if name in modelData:
                module.load_state_dict(modelData[name])

modelName = 'ResNet34_BC_lr0.01'
modelConfig = lab2_task_list.GetTasks()[modelName]
model = modelConfig.GetModel()

# origData =  torch.load(modelConfig.ckptPath)
# model.load_state_dict(origData['net'])

# Save
savePath = './ResNet34_BC.pth'
# SaveBCModel(model, modelConfig.ckptPath, savePath)

# Restore
binarizedData = torch.load(savePath)
RestoreBCModel(model, binarizedData)

# Test
model.to(device)
_, _, testloader = base.PrepareData()
criterion = nn.CrossEntropyLoss()

TestRoutine = modelConfig.GetTestRoutine(model, criterion)

base.ManuelTest(device, model, criterion, testloader, TestRoutine)