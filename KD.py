import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from models import *
from methods import GenBCModel, BCTestRoutine
import base

def train_step(
    teacher,
    student,
    train_loader,
    optimizer,
    student_loss_fn,
    divergence_loss_fn,
    temp,
    alpha,
    epoch,
    device
):
    losses = []
    pbar = tqdm(train_loader, total=len(train_loader), position=0, leave=True, desc=f"Epoch {epoch}")
    for data, targets in pbar:
        # Get data to cuda if possible
        data = data.to(device)
        targets = targets.to(device)

        # forward
        with torch.no_grad():
            teacher_preds = teacher(data)

        student.binarization()

        student_preds = student(data)
        student_loss = student_loss_fn(student_preds, targets)
        
        ditillation_loss = divergence_loss_fn(
            F.log_softmax(student_preds / temp, dim=1),
            F.log_softmax(teacher_preds / temp, dim=1)
        )
        loss = alpha * student_loss + (1 - alpha) * ditillation_loss
        losses.append(loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()

        student.restore()

        optimizer.step()
        student.clip()
    
    avg_loss = sum(losses) / len(losses)
    return avg_loss
  
def main(epochs, temp=7, alpha=0.3):
    device = base.CheckEnvironment()
    print('Prepare data....')
    trainloader, _, testloader = base.PrepareData()

    print('Gen....')
    teacher = GenBCModel(lambda:densenet_cifar()).to(device)
    print('load....')
    teacher_data = torch.load('./data/teacher/data.pth')
    teacher.load_state_dict(teacher_data['net'])

    print('Student....')
    student = GenBCModel(lambda: torch.load('./data/student/model.pth')).to(device)
    student_data = torch.load('./data/student/data.pth')
    student.load_state_dict(student_data['net'])

    student_loss_fn = nn.CrossEntropyLoss()
    divergence_loss_fn = nn.KLDivLoss(reduction="batchmean")
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)

    best_acc = 0

    teacher.eval()
    teacher.binarization()
    for epoch in range(epochs):
        student.train()
        loss = train_step(
            teacher,
            student,
            trainloader,
            optimizer,
            student_loss_fn,
            divergence_loss_fn,
            temp,
            alpha,
            epoch,
            device
        )

        acc, _ = base.ManuelTest(device, student, student_loss_fn, testloader, BCTestRoutine)
        if acc > best_acc:
            print('Saving....................................')
            torch.save(student, './DA.pth')
            best_acc = acc

if __name__ == "__main__":
    print('enter main')
    main(100)