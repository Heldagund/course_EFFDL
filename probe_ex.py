import torch
import argparse
import os

def ParseCommand():
    parser = argparse.ArgumentParser(description='Display training status')
    parser.add_argument('root', type=str,
                        help='Root directory for training')
    args = parser.parse_args()
    return args.root

def GetLastLines(fileName, lineCnt):
    fileSize =  os.path.getsize(fileName)
    with open(fileName, 'rb') as f:
        offset = -8
        lastLines = []
        while -offset < fileSize:
            f.seek(offset, 2)
            lines = f.readlines()
            if len(lines) >= lineCnt + 1:
                last_lines = lines[-lineCnt:]
                break
            offset *= 2
    return last_lines

def Probe(root):
    ckptPath = root + '/checkpoint/ckpt.pth'
    instPath = root + '/tmp/inst'

    if not os.path.exists(ckptPath):
        print('ERROR: No task list using the directory: %s'%root)
        return

    checkpoint = torch.load(ckptPath)
    print(checkpoint)
    runningList = []
    for taskName in checkpoint:
        if checkpoint[taskName] == 2:
            runningList.append(taskName)
    
    if len(runningList) == 1:
        taskName = runningList[0]
        if os.path.exists(instPath):
            print('Task %s is running...' % taskName)
            # print last record in the log file
            logPath = root + '/log/' + taskName + '.txt'
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
