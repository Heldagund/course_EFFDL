from torchvision.datasets import CIFAR10
import numpy as np
import torchvision.transforms as transforms
import torch
from torch.utils.data.dataloader import DataLoader

import os
import sys
import argparse
from datetime import datetime

from utils import progress_bar, getDuration, countParameters
from base import top_state, sub_state, dispatcher

class execution(top_state):
    def __init__(self):
        super(execution, self).__init__()

        # Substates
        self.SetSubstate(updateTaskList = update_task_list(),
                         fetchTask = fetch_task(), 
                         launchTask = launch_task(),
                         clean = clean())

    def BuildControlFlow(self):
        self.updateTaskList >= self.fetchTask >= self.launchTask >= self.updateTaskList
        self.fetchTask >= self.clean
        self.launchTask >= self.clean
        return self.updateTaskList

class update_task_list(sub_state):
    def Action(self):
        self.__UpdateTaskList()
        print('Number of tasks: ', len(self.taskList.GetTasks()))
        return self.nextState[0]

    def __UpdateTaskList(self):
        # Load the modification file and apply all of them
        if os.path.isfile(self.taskList.taskModifPath):
            modifFile = torch.load(self.taskList.taskModifPath)
            if 'import_command' in modifFile:
                for importCom in modifFile['import_command']:
                    exec(importCom)

            if 'new_task' in modifFile:
                for task_dict_str in modifFile['new_task']:
                    exec('_task_dict='+task_dict_str, locals())
                    task_dict = locals()['_task_dict']
                    self.taskList.AddTaskByDict(task_dict)
                    
            if 'delete_task' in modifFile:
                for taskName in modifFile['delete_task']:
                    if not self.taskList.DeleteTask(taskName):
                        print("WARNING: Delete task %s falied!"%taskName)

            os.remove(self.taskList.taskModifPath)

class fetch_task(sub_state):
    def Action(self):
        self.__FetchTask()
        if self.modelName=='':
            print('All tasks completed')
            nextState = self.GetStateByClass(clean)
        else:
            nextState = self.GetStateByClass(launch_task)
        return nextState

    def __FetchTask(self):
        torch.cuda.empty_cache()

        best_acc = 0
        best_epoch = -1
        if self.modelName=='':
            for key in  self.taskList.checklist:     # Get a task from the checklist
                if  self.taskList.checklist[key]==0:
                    self.modelName = key
                    break                    
            if self.modelName=='':     # All tasks have been completed
                return
            
            task = self.taskList.GetTasks()[self.modelName]
            logfile = open(self.taskList.GetTasks()[self.modelName].logPath,'w')
            print('==> Building model...')
            print('Model name: ', self.modelName)
            net = task.GetModel().to(self.device)
            # if device == 'cuda':
            #     net = torch.nn.DataParallel(net)
        else:     # Resume a given task
            task = self.taskList.GetTasks()[self.modelName]
            logfile = open(task.logPath,'a')
            netData = torch.load(task.ckptPath)
            print('==> Rebuilding model...')
            print('Model name: ', self.modelName)
            net = task.GetModel().to(self.device)
            # if device == 'cuda':
            #     net = torch.nn.DataParallel(net)
            net.load_state_dict(netData['net'])
            best_acc = netData['acc']
            best_epoch = netData['epoch']

        self.plotList['name'].append(self.modelName)

        if task.useSubset:
            activeTrainloader = self.trainloader_subset
        else:
            activeTrainloader = self.trainloader
        
        # Update data flow
        self.AddToDataflow(net = net,
                           logfile = logfile,
                           best_acc = best_acc,
                           best_epoch = best_epoch,
                           start_epoch = best_epoch+1,
                           activeTrainloader = activeTrainloader)

        # Save the initial state
        self.__SaveInitialState()

        custRoutine = task.GetCustomize
        if not custRoutine == None:
            self.AddToDataflow(customize = custRoutine)
    
    def __SaveInitialState(self):
        print('Saving...')

        # Save checklist
        self.taskList.checklist[self.modelName] = 2
        torch.save(self.taskList.checklist, self.taskList.ckptPath)

        # Save the network and its best score
        state = {'name': self.modelName,
                'net': self.net.state_dict(),
                'acc': 0,
                'epoch': 0}
        torch.save(state, self.taskList.GetTasks()[self.modelName].ckptPath)

class clean(sub_state):
    def Action(self):
        self.__Clean()
        return self.nextState[0]

    def __Clean(self):
        self.instFile.close()
        os.remove(self.taskList.instPath)
        self.resultFile.close()

class launch_task(top_state):
    def __init__(self):
        super(launch_task, self).__init__()

        # Substates
        self.SetSubstate(startTask = start_task(),
                         train = train(),
                         test = test(),
                         saveAndStep = save_and_step(), 
                         endTask = end_task())

    def BuildControlFlow(self):
        self.startTask >= self.train >= self.test >= self.saveAndStep >= self.train
        self.saveAndStep >= self.endTask
        return self.startTask
    
    def GetNextState(self, subDataflow):
        if subDataflow['isKeyInterrupt']:
            nextState = self.GetStateByClass(clean)
        else:
            nextState = self.GetStateByClass(update_task_list)
        return nextState, subDataflow

    def Action(self):       # Overload Action in order to deal with exception
        entryState = self.BuildControlFlow()
        localDispatcher = dispatcher(entryState, **self.dataflow)
        try:
            subDataflow = localDispatcher.run()
        except KeyboardInterrupt:
            subDataflow = dict(isKeyInterrupt=True)
        nextState, finalDataflow = self.GetNextState(subDataflow)
        self.AddToDataflow(**finalDataflow)
        return nextState

class start_task(sub_state):
    def Action(self):
        self.__StartTask()
        return self.nextState[0]

    def __StartTask(self):
        task = self.taskList.GetTasks()[self.modelName]
        optimizer = task.GetOptimizer(self.net)
        criterion = task.GetCriterion()

        scheduler = task.GetScheduler
        if not scheduler == None:
            scheduler = scheduler(optimizer)
            
        self.AddToDataflow(optimizer = optimizer,
                           criterion = criterion,
                           scheduler = scheduler)        
        
        if self.start_epoch == 0:
            taskData = {'loss': [], 'acc': []}
        else:     # The task is resumed so we load the previous data
            if os.path.isfile(task.dataPath):
                prevData = torch.load(task.dataPath)
                taskData = {}
                # Truncate until the best epoch
                taskData['loss'] = prevData['loss'][:self.best_epoch+1]
                taskData['acc'] = prevData['acc'][:self.best_epoch+1]
            else:
                # TODO: handle exception
                taskData = {'loss': [], 'acc': []}
        self.AddToDataflow(taskData = taskData)

        self.AddToDataflow(epoch = self.start_epoch, isKeyInterrupt = False)
        self.AddToDataflow(startTime = datetime.now())

class train(sub_state):
    def Action(self):
        self.__Train()
        return self.nextState[0]

    def __Train(self):
        print('==> Training...')
        loss = self.__Train__()
        self.taskData['loss'].append(loss)
    
    def __Train__(self):
        msg = '\nEpoch: %d' % self.epoch
        print(msg)
        self.logfile.write(msg+'\n')

        TrainRoutine = self.taskList.GetTasks()[self.modelName].GetTrainRoutine(self.net, self.optimizer, self.criterion)

        self.net.train()          # Sets the module in training mode
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, labels) in enumerate(self.activeTrainloader, 0):
        
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            outputs, loss = TrainRoutine(inputs, labels)

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            msg = 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total)
            progress_bar(batch_idx, len(self.activeTrainloader), msg)

        self.logfile.write('At the end of training\n'+msg+'\n')
        return train_loss

class test(sub_state):
    def Action(self):
        self.__Test()
        return self.nextState[0]

    def __Test(self):
        print('==> Testing...')
        acc = self.__Test__()
        self.taskData['acc'].append(acc)

        if acc > self.best_acc:
            self.best_acc = acc
            self.best_epoch = self.epoch
            self.__SaveCheckpoint()

    def __Test__(self):
        TestRoutine = self.taskList.GetTasks()[self.modelName].GetTestRoutine(self.net, self.criterion)

        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(self.testloader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
            
                outputs, loss = TestRoutine(inputs, labels)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                msg = 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total)
                progress_bar(batch_idx, len(self.testloader), msg)
            
            self.logfile.write('Test result\n'+msg+'\n')

        return 100.*correct/total
    
    def __SaveCheckpoint(self):
            print('Saving...')

            # Save checklist
            self.taskList.checklist[self.modelName] = 2
            torch.save(self.taskList.checklist, self.taskList.ckptPath)

            # Save the network and its best score
            state = {'name': self.modelName,
                    'net': self.net.state_dict(),
                    'acc': self.best_acc,
                    'epoch': self.best_epoch}
            torch.save(state, self.taskList.GetTasks()[self.modelName].ckptPath)

class save_and_step(sub_state):
    def Action(self):
        self.__SaveAndStep()
        if self.epoch > 200 + self.start_epoch:
            nextState = self.GetStateByClass(end_task)
        else:
            nextState = self.GetStateByClass(train)
        return nextState

    def __SaveAndStep(self):
        msg = 'Best accuracy: %.3f%% at epoch %d' % (self.best_acc, self.best_epoch)
        print(msg)

        elapsedTime = getDuration(self.startTime, datetime.now())
        print(elapsedTime)

        self.logfile.write(elapsedTime+'\n'+msg+'\n')
        self.logfile.flush()       # Avoid losing results

        if not self.scheduler == None:
            self.scheduler.step()
            
        self.epoch += 1
    
        # Save the average loss and the accuracy achived at each epoch
        torch.save(self.taskData, self.taskList.GetTasks()[self.modelName].dataPath)

        self.AddToDataflow(msg=msg, elapsedTime=elapsedTime)

class end_task(sub_state):
    def Action(self):
        self.__EndTask()
        return self.nextState[0]

    def __EndTask(self):
        # Update and save the checklist
        self.taskList.checklist[self.modelName] = 1
        torch.save(self.taskList.checklist, self.taskList.ckptPath)

        # Save the best accuracy and the number of parameters of the network
        self.plotList['accuracy'].append(self.best_acc)
        parameterCnt = countParameters(self.net)
        self.plotList['parameterCnt'].append(parameterCnt)
        torch.save(self.plotList, self.taskList.plotDataPath)

        # Save the result
        self.resultFile.write('{}:\nNumber of parameters: {}\n{}\n{}\n\n'.format(self.modelName, parameterCnt, self.elapsedTime, self.msg))
        self.logfile.close()
        self.modelName = ''