import base
from tasks import pretrain_task_list

def main():
    isResume = base.ParseCommand()
    trainloader, trainloader_subset, testloader = base.PrepareData()
    device = base.CheckEnvironment()
    base.RunTasks(isResume, device, trainloader, trainloader_subset, testloader, pretrain_task_list)
    base.PlotResult(pretrain_task_list)
    base.SendResult(pretrain_task_list)

if __name__ == '__main__':
    main()
