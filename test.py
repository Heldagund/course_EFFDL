# import torch
# import base
# from methods import BCTestRoutine

# model = torch.load('./DA.pth')

# _, _, testloader = base.PrepareData()
# device = base.CheckEnvironment()

# criterion =  torch.nn.CrossEntropyLoss()

# base.ManuelTest(device, model, criterion, testloader, BCTestRoutine)

import torch
import base
from utils import progress_bar

_, _, testloader = base.PrepareData()
device = base.CheckEnvironment()

model = torch.load('./grouped.pth')

data = torch.load('./../EFFDL_Data/checkpoint/DenseNetCifar_lr0.01_pruned_grouped.pth')

model.load_state_dict(data['net'])

criterion = torch.nn.CrossEntropyLoss()

model.eval()

model.half().to(device)
test_loss = 0
correct = 0
total = 0
with torch.no_grad():
    for batch_idx, (inputs, labels) in enumerate(testloader):
        inputs, labels = inputs.half().to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        msg = 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total)
        progress_bar(batch_idx, len(testloader), msg)
