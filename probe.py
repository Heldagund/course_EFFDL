import torch
import argparse
import os

from tasks import *
from utils import GetLastLines

def ParseCommand():
    parser = argparse.ArgumentParser(description='Display training status')
    parser.add_argument('task_list', type=str,
                        help='Name of the task list')
    args = parser.parse_args()
    return args.task_list

def Probe(taskListStr):
    try:
        taskList = globals()[taskListStr]
    except KeyError as ke:
        print('ERROR: No task list named ', ke)
        return

    checkpoint = torch.load(taskList.ckptPath)
    print(checkpoint)
    runningList = []
    for taskName in checkpoint:
        if checkpoint[taskName] == 2:
            runningList.append(taskName)
    
    if len(runningList) == 1:
        taskName = runningList[0]
        if os.path.exists(taskList.instPath):
            print('Task %s is running...' % taskName)
            # print last record in the log file
            if taskName in taskList.GetTasks():       # Newly added task is not in the old task list
                logPath = taskList.GetTasks()[taskName].logPath
            else:
                logPath = taskList.root + '/log/' + taskName + '.txt'

            lastRecord = GetLastLines(logPath, 7)
            print('Last record:')
            for line in lastRecord:
                print('  ', line.decode(), end="")
        else:
            print('No task is running...')
            print('Task %s can be resumed' % taskName)

    elif len(runningList) == 0:
        print('No task is running...')
        
    else:
        print('UNEXPECTED: Some of these tasks could have been terminated before but not yet completed !')
        for taskName in runningList:
            print('  ', taskName)

if __name__ == '__main__':
    Probe(ParseCommand())
