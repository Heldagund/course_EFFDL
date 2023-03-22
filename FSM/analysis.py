import matplotlib.pyplot as plt
import numpy as np
import torch

from base import top_state, sub_state
from utils import sendMail

class analysis(top_state):
    def __init__(self):
        super(analysis, self).__init__()

        # Substates
        self.SetSubstate(plotResult = plot_result(), 
                         sendResult = send_result())
    
    def BuildControlFlow(self):
        self.plotResult >= self.sendResult
        return self.plotResult

class plot_result(sub_state):
    def Action(self):
        self.__PlotResult()
        return self.nextState[0]
    
    def __PlotResult(self):
        # Plot the number of parameters and the best accuracy
        plotList = torch.load(self.taskList.plotDataPath)
        print(plotList)
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

        fig.savefig(self.taskList.plotFigurePath)

        # Plot the loss and accuracy evolution for every task
        task_dict = self.taskList.GetTasks()
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
            fig_loss.savefig(self.taskList.root + '/result/' + modelName + '_loss.png')

            fig_acc = plt.figure()
            plt.plot(x, acc)
            plt.xlabel("Number of epoch")
            plt.ylabel("Accuracy")
            plt.title("Accuracy increasing during traing")
            fig_acc.savefig(self.taskList.root + '/result/' + modelName + '_acc.png')
            plt.close()

class send_result(sub_state):
    def Action(self):
        self.__SendResult()
        return self.nextState[0]
    
    def __SendResult(self):
        # Send email
        message = 'All tasks completed. See attached file for results.'
        subject = 'EFFDL news'
        sender = 'Heldagund'
        recipient = 'related'
        attachments = self.taskList.root + '/result'
            
        to_addrs = 'heldagund@gmail.com,kehanliu2000@163.com'
        sendMail(subject, message, sender, recipient, to_addrs, attachments)