import numpy as np
import random
import torch

import base
from utils import getDuration

def Mixup(inputs, labels):
    lam = random.random()
    dev = torch.device(inputs.device)

    indexPerm = torch.randperm(inputs.size(0))
    indexPerm = indexPerm.to(dev)

    inputsPerm = torch.index_select(inputs, 0, indexPerm)
    labelsPerm = torch.index_select(labels, 0, indexPerm)

    inputsMixed = lam*inputs + (1-lam)*inputsPerm
    
    return lam, inputsMixed, labelsPerm

def MixupTrainRoutine(net, optimizer, criterion, inputs, labels):
    optimizer.zero_grad()

    lam, inputsMixed, labelsPerm = Mixup(inputs, labels)

    outputs = net(inputsMixed)    
    loss = lam*criterion(outputs, labels) + (1-lam)*criterion(outputs, labelsPerm)
    loss.backward()

    optimizer.step()
    
    return outputs, loss

def MixupWarmStart(device, trainloader, testloader, model_dict, taskList):
    # Read the model_dict
    net = model_dict['net']
    modelName = model_dict['name']
    logfile = model_dict['logfile']
    best_acc = model_dict['best_acc']
    best_epoch = model_dict['best_epoch']
    start_epoch = model_dict['start_epoch']

    optimizer = taskList.GetTasks()[modelName].GetOptimizer(net)
    criterion = taskList.GetTasks()[modelName].GetCriterion()
    scheduler = taskList.GetTasks()[modelName].GetScheduler(optimizer)

    if start_epoch == 0:
        taskData = {'loss': [], 'acc': []}
    else:     # The task is resumed so we load the previous data
        if os.path.isfile(taskList.GetTasks()[modelName].dataPath):
            prevData = torch.load(taskList.GetTasks()[modelName].dataPath)
            taskData = {}
            # Truncate until the best epoch
            taskData['loss'] = prevData['loss'][:best_epoch]
            taskData['acc'] = prevData['acc'][:best_epoch]
        else:
            # TODO: handle exception
            taskData = {'loss': [], 'acc': []}

    startTime = datetime.now() 
    for epoch in range(start_epoch, 200):
        msg = '\nEpoch: %d' % epoch
        print(msg)
        logfile.write(msg+'\n')

        print('==> Training...')
        if epoch < 20:
            loss, msg = base.ManuelTrain(device, net, optimizer, criterion, trainloader, base.DefaultTrainRoutine)
        else:
            loss, msg = base.ManuelTrain(device, net, optimizer, criterion, trainloader, MixupTrainRoutine)
            
        logfile.write('At the end of training\n'+msg+'\n')
        
        taskData['loss'].append(loss)

        print('==> Testing...')
        acc = base.Test(device, model_dict, criterion, testloader, taskList)
        taskData['acc'].append(acc)

        if acc > best_acc:
            base.SaveCheckpoint(modelName, net, acc, epoch, taskList)
            best_acc = acc
            best_epoch = epoch

        msg = 'Best accuracy: %.3f%% at epoch %d' % (best_acc, best_epoch)
        print(msg)

        elapsedTime = getDuration(startTime, datetime.now())
        print(elapsedTime)

        logfile.write(elapsedTime+'\n'+msg+'\n')
        logfile.flush()       # Avoid losing results

        scheduler.step()
    
        # Save the average loss and the accuracy achived at each epoch
        torch.save(taskData, taskList.GetTasks()[modelName].dataPath)

    logfile.close()

    return best_acc, elapsedTime, msg