import argparse
import os
import torch

from tasks import *

def ParseCommand():
    parser = argparse.ArgumentParser(description='Display training status')
    parser.add_argument('task_list', type=str,
                        help='Name of the task list')
    args = parser.parse_args()
    return args.task_list

def AddImportCommand(importCom, modifDict):
    if len(importCom) > 0:
        try:
            exec(importCom)
        except:
            print('ERROR: Invalid import command')
            return False

        if 'import_command' in modifDict:
            modifDict['import_command'].append(importCom)
        else:
            modifDict['import_command'] = [importCom]
        
        return True

def AddNewTask(task_dict_str, modifDict):
    if 'new_task' in modifDict:
        modifDict['new_task'].append(task_dict_str)
    else:
        modifDict['new_task'] = [task_dict_str]

def AddDeleteTask(taskName, modifDict):
    if 'delete_task' in modifDict:
        modifDict['delete_task'].append(taskName)
    else:
        modifDict['delete_task'] = [taskName]

def AddTask(taskListStr, task_dict_str, importCom=''):
    try:
        taskList = globals()[taskListStr]
    except KeyError as ke:
        print('No task list named ', ke)
        return
    
    if os.path.isfile(taskList.instPath):
        # Update the task list
        if os.path.isfile(taskList.taskModifPath):
            modifDict = torch.load(taskList.taskModifPath)
        else:
            modifDict = {}
        
        AddNewTask(task_dict_str, modifDict)
        if not AddImportCommand(importCom, modifDict):
            return

        torch.save(modifDict, taskList.taskModifPath)
        print('SUCCESS')
    else:
        print('ERROR: The given task list is not being executed')

def DeleteTask(taskListStr, taskName):
    try:
        taskList = globals()[taskListStr]
    except KeyError as ke:
        print('No task list named ', ke)
        return
    
    if taskName not in taskList.GetTasks():
        print('No task named ', taskName)
        return

    if os.path.isfile(taskList.instPath):
        # Update the task list
        if os.path.isfile(taskList.taskModifPath):
            modifDict = torch.load(taskList.taskModifPath)
        else:
            modifDict = {}
        
        AddDeleteTask(taskName, modifDict)

        torch.save(modifDict, taskList.taskModifPath)
        print('SUCCESS')
    else:
        print('ERROR: The given task list is not being executed')

if __name__ == '__main__':
    taskListStr = ParseCommand()
    DeleteTask(taskListStr, 'densenet_cifar_pruned2_BC_lr0.01')
    # AddTask(taskListStr, '''{
    #                         "name": "DenseNetCifar_BC_lr0.01",
    #                         "model": lambda: GenBCModel(lambda:densenet_cifar()),
    #                         "optimizer": lambda net:optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4),
    #                         "train_routine": BCTrainRoutine,
    #                         "test_routine": BCTestRoutine
    #                         }''',
    #                        '''from models import densenet_cifar
    #                           \nimport torch.optim as optim
    #                           \nfrom methods import GenBCModel, BCTrainRoutine, BCTestRoutine''')
