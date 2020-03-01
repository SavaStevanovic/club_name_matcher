import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-12

    def forward(self, output1, output2, label):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = label * distances + (1 + -1 * label) * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2)
        return losses

def accuracy( net, dataloader, epoch=1):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            inputs1, inputs2, loss_mul, labels = data['first_text'].to('cuda'), data['second_text'].to('cuda'), data['loss_mul'].to('cuda'), data['label'].to('cuda')
            outputs1, outputs2 = net(inputs1, inputs2)
            loss = (outputs2 - outputs1).pow(2).sum(1)
            predicted = loss < 0.5
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    acc = 100 * correct / total
    return acc

def fit_epoch(net, trainloader, writer, epoch=1):
    net.train()
    optimizer = optim.Adam(net.parameters())
    criterion = ContrastiveLoss()
    loss = 0.0

    for i, data in enumerate(tqdm(trainloader)):

        # get the inputs; data is a list of [inputs, labels]
        inputs1, inputs2, loss_mul, labels = data['first_text'].to('cuda'), data['second_text'].to('cuda'), data['loss_mul'].to('cuda'), data['label'].to('cuda')

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs1, outputs2 = net(inputs1, inputs2)
        loss = criterion(outputs1, outputs2, labels)
        loss = torch.dot(loss, loss_mul)
        loss.backward()
        optimizer.step()

        loss += loss.item()

    # ...log the running loss
    writer.add_scalar('training loss', loss/len(trainloader), epoch)

def validate_epoch(net, validationloader, writer, epoch=1):
    net.eval()
    criterion = ContrastiveLoss()
    loss = 0.0

    with torch.no_grad():
        for i, data in enumerate(tqdm(validationloader)):
            # get the inputs; data is a list of [inputs, labels]
            inputs1, inputs2, loss_mul, labels = data['first_text'].to('cuda'), data['second_text'].to('cuda'), data['loss_mul'].to('cuda'), data['label'].to('cuda')

            # forward + backward + optimize
            outputs1, outputs2 = net(inputs1, inputs2)
            loss = criterion(outputs1, outputs2, labels)
            loss = torch.dot(loss, loss_mul)
            loss += loss.item()

    # ...log the running loss
    val_loss = loss/len(validationloader)
    writer.add_scalar('validation loss', val_loss, epoch)

    return val_loss

def fit(net, trainloader, validationloader, epochs=1000):
    log_datatime = str(datetime.now().time())
    writer = SummaryWriter(os.path.join('logs', log_datatime))
    best_acc = 0
    for epoch in range(epochs):
        fit_epoch(net, trainloader, writer, epoch=epoch)
        val_loss = validate_epoch(net, validationloader, writer, epoch)
        # train_acc = accuracy(net, validationloader, epoch)
        val_acc = accuracy(net, trainloader, epoch)
        writer.add_scalars('acc', {'val_acc':val_acc}, epoch)
        if best_acc < val_acc:
            best_acc = val_acc
            print('Saving model with acc: {}'.format(val_acc))
            chp_dir = 'checkpoints'
            os.makedirs((chp_dir), exist_ok=True)
            torch.save(net, os.path.join(chp_dir, 'checkpoints.pth'))
        else:
            print('Epoch {} acc: {}'.format(epoch, val_acc))
    print('Finished Training')