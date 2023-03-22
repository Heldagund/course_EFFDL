import base
from tasks import lab1_task_list

def main():
    isResume = base.ParseCommand()
    trainloader, trainloader_subset, testloader = base.PrepareData()
    device = base.CheckEnvironment()
    base.RunTasks(isResume, device, trainloader, trainloader_subset, testloader, lab1_task_list)
    # base.PlotResult(lab1_task_list)
    # base.SendResult(lab1_task_list)

if __name__ == '__main__':
    main()