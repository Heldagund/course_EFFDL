import torch
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
import numpy as np
import argparse

from base import top_state, sub_state

class initialization(top_state):
    def __init__(self):
        super(initialization, self).__init__()

        # Substates
        self.SetSubstate(parseCommand = parse_command(), 
                         prepareData = prepare_data(),
                         setupEnvironment = setup_environment())

    def BuildControlFlow(self):
        self.parseCommand >= self.prepareData >= self.setupEnvironment
        return self.parseCommand

class parse_command(sub_state):
    def Action(self):
        self.__ParseCommand()
        return self.nextState[0]

    def __ParseCommand(self):
        parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
        parser.add_argument('--resume', '-r', action='store_true',
                            help='resume from checkpoint')
        args = parser.parse_args()
        self.AddToDataflow(needResume = args.resume)

class prepare_data(sub_state):
    def Action(self):
        self.__PrepareData()
        return self.nextState[0]

    def __PrepareData(self):
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
        self.AddToDataflow(trainloader=trainloader, trainloader_subset=trainloader_subset, testloader=testloader)

class setup_environment(sub_state):
    def Action(self):
        self.__SetupEnvironment()
        return self.nextState[0]

    def __SetupEnvironment(self):
        # Check if GPU is available
        device = self.__CheckDevice()

        if self.needResume:
            # Load checkpoint
            modelName = self.__LoadCheckpoint()
            resultFile = open(self.taskList.resultTextPath,'a')
        else:
            modelName = ''
            resultFile = open(self.taskList.resultTextPath,'w')

        # Create a file to receive instruction
        instFile = open(self.taskList.instPath, 'wb+')
        
        # Create a dictionary to save results for plotting
        plotList = {'name': [], 'accuracy': [], 'parameterCnt': []}

        self.AddToDataflow(device = device,
                           modelName = modelName,
                           resultFile = resultFile,
                           instFile = instFile,
                           plotList = plotList)
    
    def __CheckDevice(self):
        isCudaAvailable = torch.cuda.is_available()
        print('Cuda is available: ', isCudaAvailable)

        if isCudaAvailable:
            torch.backends.cudnn.benchmark = True
            return 'cuda'
        else:
            return 'cpu'
    
    def __LoadCheckpoint(self):
        print('==> Resuming from checkpoint...')
        checkpoint = torch.load(self.taskList.ckptPath)
        modelName = ''
        ckptNeedUpdate = False 
        for taskName in self.taskList.GetTasks():
            if taskName in checkpoint:
                if checkpoint[taskName]==2:      # The task is in progress
                    modelName = taskName
                self.taskList.checklist[taskName] = checkpoint[taskName]     # Restore checklist from checkpoint
            else:                                # New task is added
                checkpoint[taskName] = 0
                ckptNeedUpdate = True
        if ckptNeedUpdate:
            torch.save(checkpoint, self.taskList.ckptPath)
        return modelName
