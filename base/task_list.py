import torch.nn as nn
import torch.optim as optim
import os

defaultCriterion = lambda:nn.CrossEntropyLoss()
defaultOptimizer = lambda net:optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
defaultScheduler = lambda optimizer:optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

def DefaultTrainRoutine(net, optimizer, criterion, inputs, labels):
    optimizer.zero_grad()

    outputs = net(inputs)

    loss = criterion(outputs, labels)
    loss.backward()

    optimizer.step()
    return outputs, loss

def DefaultTestRoutine(net, criterion, inputs, labels):
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    return outputs, loss

'''
task parameters:
    required: name, model
    optional: criterion, optimizer, scheduler, train_routine, task_routine, customize

    format of train_routine: see DefaultTrainRoutine

    format of test_routine: see DefaultTestRoutine    

    format of customize(customized training method):
        best_acc, elapsedTime, lastMsg = cust(device, activeTrainloader, testloader, model_dict, _taskList)
'''
class task:
    def __init__(self, task_dict):
        self.name = task_dict['name']
        self.GetModel = task_dict['model']        # Funciton object to get untrained model
        if 'criterion' in task_dict:
            self.GetCriterion = task_dict['criterion']
        else:
            self.GetCriterion = defaultCriterion

        if 'optimizer' in task_dict:
            self.GetOptimizer = task_dict['optimizer']
        else:
            self.GetOptimizer = defaultOptimizer
        
        if 'scheduler' in task_dict:
            self.GetScheduler = task_dict['scheduler']
        else:
            self.GetScheduler = defaultScheduler
        
        if 'train_routine' in task_dict:
            self.GetTrainRoutine = self.__GenTrainRoutine(task_dict['train_routine'])
        else:
            self.GetTrainRoutine = self.__GenTrainRoutine(DefaultTrainRoutine)
        
        if 'test_routine' in task_dict:
            self.GetTestRoutine = self.__GenTestRoutine(task_dict['test_routine'])
        else:
            self.GetTestRoutine = self.__GenTestRoutine(DefaultTestRoutine)
        
        if 'customize' in task_dict:
            self.GetCustomize = task_dict['customize']
        else:
            self.GetCustomize = None

        if 'useSubset' in task_dict:
            self.useSubset = task_dict['useSubset']
        else:
            self.useSubset = False

        self.logPath = ''
        self.ckptPath = ''
        self.dataPath = ''      # Data used to plot
    
    def genPaths(self, root):
        self.logPath = root + '/log/' + self.name + '.txt'
        self.ckptPath = root + '/checkpoint/' + self.name + '.pth'
        self.dataPath = root + '/data/' + self.name + '.pth'
    
    def __CurryTrainRoutine(self, net, optimizer, criterion, trainRoutine):     # Curring
        return lambda inputs, labels: trainRoutine(net, optimizer, criterion, inputs, labels)

    def __GenTrainRoutine(self, trainRoutine):
        return lambda net, optimizer, criterion: self.__CurryTrainRoutine(net, optimizer, criterion, trainRoutine)

    def __CurryTestRoutine(self, net, criterion, testRoutine):     # Curring
        return lambda inputs, labels: testRoutine(net, criterion, inputs, labels)

    def __GenTestRoutine(self, testRoutine):
        return lambda net, criterion: self.__CurryTestRoutine(net, criterion, testRoutine)

'''
task_list parameters:
    required: root
'''

class task_list:
    def __init__(self, root):
        self.tasks_dict = {}
        self.root = root

        # make directories
        if not os.path.isdir(root):
            os.mkdir(root)
        if not os.path.isdir(root + '/log'):
            os.mkdir(root + '/log')
        if not os.path.isdir(root + '/data'):
            os.mkdir(root + '/data')
        if not os.path.isdir(root + '/checkpoint'):
            os.mkdir(root + '/checkpoint')
        if not os.path.isdir(root + '/result'):
            os.mkdir(root + '/result')
        if not os.path.isdir(root + '/tmp'):
            os.mkdir(root + '/tmp')
        
        self.checklist = {}
        
        self.ckptPath = root + '/checkpoint/ckpt.pth'
        self.resultTextPath = root + '/result/result.txt'
        self.plotDataPath = root + '/result/plot_data.pth'
        self.plotFigurePath = root + '/result/result.png'
        self.taskModifPath = root + '/tmp/task_modification'
        self.instPath = root + '/tmp/inst'

    def AddTask(self,task):
        self.tasks_dict[task.name] = task       
        task.genPaths(self.root)

        self.checklist[task.name] = 0   # 0:to do     1:completed      2:running
    
    def AddTaskByDict(self,task_dict):
        newTask = task(task_dict)
        self.AddTask(newTask)    
        
    
    def DeleteTask(self, taskName):
        if taskName in self.tasks_dict:
            if self.checklist[taskName] == 0:
                del self.tasks_dict[taskName]
                del self.checklist[taskName]
                return True
            else:
                print('Cannot delete a task done or still running')
        else:
            print('No task named ', taskName)

        return False

    def GetTasks(self):
        return self.tasks_dict

# Generate a tasklist from a configuration list for parameter sweep
def GenTaskList(config_list, root):
    tl = task_list(root)
    td_list = GenTaskDictList(config_list)
    for td in td_list:
        tl.AddTask(task(td))
    return tl

def GenTaskDictList(config_list):
    if len(config_list) == 1:
        lastParam = config_list[0]['param']
        lastOptions = config_list[0]['options']
        td_list = []
        for option in lastOptions:
            td = {}
            td['name'] = option['tag']
            td[lastParam] = option['value']
            td_list.append(td)
        return td_list
    else:
        td_list = []
        firstParam = config_list[0]['param']
        firstOptions = config_list[0]['options']
        for task_dict in GenTaskDictList(config_list[1:]):
            for option in firstOptions:
                td = task_dict.copy()
                td['name'] = td['name'] + '_' + option['tag']
                td[firstParam] = option['value']
                td_list.append(td)
        return td_list
