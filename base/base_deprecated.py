from torchvision.datasets import CIFAR10
import numpy as np
import torchvision.transforms as transforms
import torch
from torch.utils.data.dataloader import DataLoader

import os
import sys
import argparse
from datetime import datetime

from utils import progress_bar, getDuration, countParameters, sendMail
from base import DefaultTestRoutine, DefaultTrainRoutine
 
def ParseCommand():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    args = parser.parse_args()
    return args.resume

def PrepareData():
    ## Normalization adapted for CIFAR10
    normalize_scratch = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    # Transforms is a list of transformations applied on the 'raw' dataset before the data is fed to the network. 
    # Here, Data augmentation (RandomCrop and Horizontal Flip) are applied to each batch, differently at each epoch, on the training set data only
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize_scratch,
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize_scratch,
    ])

    ### The data from CIFAR10 will be downloaded in the following folder
    rootdir = '/users/local'

    c10train = CIFAR10(rootdir,train=True,download=True,transform=transform_train)
    c10test = CIFAR10(rootdir,train=False,download=True,transform=transform_test)

    trainloader = DataLoader(c10train,batch_size=32,shuffle=True)
    testloader = DataLoader(c10test,batch_size=32)

    ## number of target samples for the final dataset
    num_train_examples = len(c10train)
    num_samples_subset = 15000

    ## We set a seed manually so as to reproduce the results easily
    seed  = 2147483647

    ## Generate a list of shuffled indices ; with the fixed seed, the permutation will always be the same, for reproducibility
    indices = list(range(num_train_examples))
    np.random.RandomState(seed=seed).shuffle(indices)## modifies the list in place

    ## We define the Subset using the generated indices 
    c10train_subset = torch.utils.data.Subset(c10train,indices[:num_samples_subset])
    print(f"Initial CIFAR10 dataset has {len(c10train)} samples")
    print(f"Subset of CIFAR10 dataset has {len(c10train_subset)} samples")

    # Finally we can define anoter dataloader for the training data
    trainloader_subset = DataLoader(c10train_subset,batch_size=32,shuffle=True)
    ### You can now use either trainloader (full CIFAR10) or trainloader_subset (subset of CIFAR10) to train your networks.
    return trainloader, trainloader_subset, testloader

def CheckEnvironment():
    isCudaAvailable = torch.cuda.is_available()
    print('Cuda is available: ', isCudaAvailable)

    if isCudaAvailable:
        torch.backends.cudnn.benchmark = True

    return 'cuda' if isCudaAvailable else 'cpu' 

def LoadCheckpoint(taskList):
    print('==> Resuming from checkpoint...')
    checkpoint = torch.load(taskList.ckptPath)
    modelName = ''
    for taskName in taskList.GetTasks():
        if checkpoint[taskName]==2:      # The task is in progress
            modelName = taskName
        taskList.checklist[taskName] = checkpoint[taskName]     # Restore checklist from checkpoint
    return modelName

def BuildTask(device, modelName, taskList):
    model_dict = {'best_acc': 0, 'best_epoch': 0, 'start_epoch': 0}

    if modelName=='':
            for key in  taskList.checklist:     # Get a task from the checklist
                if  taskList.checklist[key]==0:
                    modelName = key
                    break                    
            if modelName=='':     # All tasks have been completed
                return model_dict
            logfile = open(taskList.GetTasks()[modelName].logPath,'w')
            print('==> Building model...')
            print('Model name: ', modelName)
            net = taskList.GetTasks()[modelName].GetModel().to(device)
            # if device == 'cuda':
            #     net = torch.nn.DataParallel(net)
    else:     # Resume a given task
        logfile = open(taskList.GetTasks()[modelName].logPath,'a')
        netData = torch.load(taskList.GetTasks()[modelName].ckptPath)
        print('==> Rebuilding model...')
        print('Model name: ', modelName)
        net = taskList.GetTasks()[modelName].GetModel().to(device)
        # if device == 'cuda':
        #     net = torch.nn.DataParallel(net)
        net.load_state_dict(netData['net'])
        model_dict['best_acc'] = netData['acc']
        model_dict['best_epoch'] = netData['epoch']
        model_dict['start_epoch'] = netData['epoch'] + 1
    
    # Save the initial state
    SaveCheckpoint(modelName, net, 0, 0, taskList)
    
    # Write the model dictionary
    model_dict['name'] = modelName
    model_dict['net'] = net
    model_dict['logfile'] = logfile
    custRoutine = taskList.GetTasks()[modelName].GetCustomize
    if not custRoutine == None:
        model_dict['customize'] = custRoutine
    
    return model_dict

def Train(device, model_dict, optimizer, criterion, trainloader, taskList):
    # Read the model_dict
    net = model_dict['net']
    modelName = model_dict['name']
    logfile = model_dict['logfile']
  
    TrainRoutine = taskList.GetTasks()[modelName].GetTrainRoutine(net, optimizer, criterion)

    net.train()          # Sets the module in training mode
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, labels) in enumerate(trainloader, 0):
       
        inputs, labels = inputs.to(device), labels.to(device)
        
        outputs, loss = TrainRoutine(inputs, labels)

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        msg = 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total)
        progress_bar(batch_idx, len(trainloader), msg)

    logfile.write('At the end of training\n'+msg+'\n')
    return train_loss

def Test(device, model_dict, criterion, testloader, taskList):
    # Read the model_dict
    net = model_dict['net']
    modelName = model_dict['name']
    logfile = model_dict['logfile']
  
    TestRoutine = taskList.GetTasks()[modelName].GetTestRoutine(net, criterion)

    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(testloader):
            inputs, labels = inputs.to(device), labels.to(device)
           
            outputs, loss = TestRoutine(inputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            msg = 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total)
            progress_bar(batch_idx, len(testloader), msg)
        
        logfile.write('Test result\n'+msg+'\n')

    return 100.*correct/total

def SaveCheckpoint(modelName, net, acc, epoch, taskList):
    print('Saving...')

    # Save checklist
    taskList.checklist[modelName] = 2
    torch.save(taskList.checklist, taskList.ckptPath)

    # Save the network and its best score
    state = {'name': modelName,
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch}
    torch.save(state, taskList.GetTasks()[modelName].ckptPath)

def UpdateTaskList(taskList):
    # Load the modification file and apply all of them
    if os.path.isfile(taskList.taskModifPath):
        modifFile = torch.load(taskList.taskModifPath)
        if 'import_command' in modifFile:
            for importCom in modifFile['import_command']:
                exec(importCom)

        if 'new_task' in modifFile:
            for task_dict_str in modifFile['new_task']:
                exec('_task_dict='+task_dict_str, locals())
                task_dict = locals()['_task_dict']
                taskList.AddTaskByDict(task_dict)
                
        if 'delete_task' in modifFile:
            for taskName in modifFile['delete_task']:
                if not taskList.DeleteTask(taskName):
                    print("WARNING: Delete task %s falied!"%taskName)

        os.remove(taskList.taskModifPath)  

def LaunchTask(device, trainloader, testloader, model_dict, taskList):
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
        loss = Train(device, model_dict, optimizer, criterion, trainloader, taskList)
        taskData['loss'].append(loss)

        print('==> Testing...')
        acc = Test(device, model_dict, criterion, testloader, taskList)
        taskData['acc'].append(acc)

        if acc > best_acc:
            SaveCheckpoint(modelName, net, acc, epoch, taskList)
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

def Clean(instFile, resultFile, taskList):
    instFile.close()
    os.remove(taskList.instPath)
    resultFile.close()

def RunTasks(isResume, device, trainloader, trainloader_subset, testloader, taskList):
    if isResume:
        # Load checkpoint
        modelName = LoadCheckpoint(taskList)
        resultFile = open(taskList.resultTextPath,'a')
    else:
        modelName = ''
        resultFile = open(taskList.resultTextPath,'w')

    # Create a file to receive instruction
    instFile = open(taskList.instPath, 'wb+')
    
    # Initialize a dictionary for saving results for plotting
    plotList = {'name': [], 'accuracy': [], 'parameterCnt': []}
    
    # Iterate and execute all tasks
    while True:
        torch.cuda.empty_cache()

        # Update the task list
        UpdateTaskList(taskList)
        print('Number of tasks: ', len(taskList.GetTasks()))

        # Fetch a task from tasks or resume it if modelName is not empty
        model_dict = BuildTask(device, modelName, taskList)

        if 'name' not in model_dict:     # All tasks have been completed
            break
        else:
            modelName = model_dict['name']

        # Save task name
        plotList['name'].append(modelName)

        # Choose trainloader
        if taskList.GetTasks()[modelName].useSubset:
            activeTrainloader = trainloader_subset 
        else:
            activeTrainloader = trainloader

        # Launch training
        if 'customize' in model_dict:
            print('Customized training...')
            custRoutine = model_dict['customize']
            try:
                best_acc, elapsedTime, lastMsg = custRoutine(device, activeTrainloader, testloader, model_dict, taskList)
            except KeyboardInterrupt:
                Clean(instFile, resultFile, taskList)
                return
        else:
            try:
                best_acc, elapsedTime, lastMsg = LaunchTask(device, activeTrainloader, testloader, model_dict, taskList)
            except KeyboardInterrupt:
                Clean(instFile, resultFile, taskList)
                return


        # Update and save the checklist
        taskList.checklist[modelName] = 1
        torch.save(taskList.checklist, taskList.ckptPath)

        # Save the best accuracy and the number of parameters of the network
        plotList['accuracy'].append(best_acc)
        parameterCnt = countParameters(model_dict['net'])
        plotList['parameterCnt'].append(parameterCnt)
        torch.save(plotList, taskList.plotDataPath)

        # Save the result
        resultFile.write(modelName+':\nNumber of parameters: %d\n'%(parameterCnt)+elapsedTime+'\n'+lastMsg+'\n\n')

        # Empty modelName to take the next one
        modelName = ''
    
    print('All tasks completed')

    Clean(instFile, resultFile, taskList)
        
def PlotResult(taskList):
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Plot the number of parameters and the best accuracy
    plotList = torch.load(taskList.plotDataPath)
    x = plotList['parameterCnt']
    y = plotList['accuracy']
    name = plotList['name']

    fig = plt.figure()
    plt.scatter(x, y)
    print(x)
    #plt.yticks(range(70, 100, 10))

    plt.xlabel("Number of model parameters")
    plt.ylabel("Top 1 Accuracy(%)")

    plt.title("Image Classification task on ImageNet dataset")

    for i in range(len(x)):
        plt.text(
            x[i]*1.01,
            y[i]*1.01,
            name[i],
            fontsize=10,
            color="r",
            style="italic",
            weight="light",
            verticalalignment="center",
            horizontalalignment="right",
            rotation=0
        )

    fig.savefig(taskList.plotFigurePath)

    # Plot the loss and accuracy evolution for every task
    task_dict = taskList.GetTasks()
    for modelName in task_dict:
        taskData = torch.load(task_dict[modelName].dataPath)

        loss = taskData['loss']
        acc = taskData['acc']
        x = range(0, len(loss))

        fig_loss = plt.figure()
        plt.plot(x, loss)
        plt.xlabel("Number of epoch")
        plt.ylabel("Average loss")
        plt.title("Loss descending during traing")
        fig_loss.savefig(taskList.root + '/result/' + modelName + '_loss.png')

        fig_acc = plt.figure()
        plt.plot(x, acc)
        plt.xlabel("Number of epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy increasing during traing")
        fig_acc.savefig(taskList.root + '/result/' + modelName + '_acc.png')

def SendResult(taskList):
    # Send email
    message = 'All tasks completed. See attached file for results.'
    subject = 'EFFDL news'
    sender = 'Heldagund'
    recipient = 'related'
    attachments = taskList.root + '/result'
        
    to_addrs = 'heldagund@gmail.com,kehanliu2000@163.com'
    sendMail(subject, message, sender, recipient, to_addrs, attachments)

def ManuelTest(device, model, criterion, testloader, TestRoutine = DefaultTestRoutine):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(testloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, loss = TestRoutine(model, criterion, inputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            msg = 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total)
            progress_bar(batch_idx, len(testloader), msg)
    return 100.*correct/total, msg

def ManuelTrain(device, model, optimizer, criterion, trainloader, TrainRoutine = DefaultTrainRoutine):
    model.train()          #Sets the module in training mode
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, labels) in enumerate(trainloader, 0):

        inputs, labels = inputs.to(device), labels.to(device)
        
        outputs, loss = TrainRoutine(model, criterion, inputs, labels)

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        msg = 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total)
        progress_bar(batch_idx, len(trainloader), msg)
        return train_loss, msg