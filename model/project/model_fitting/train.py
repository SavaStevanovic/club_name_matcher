import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from model_fitting.evaluate import accuracy
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

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True).view(-1)
        loss_contrastive = (1-label) * torch.pow(euclidean_distance, 2) + label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)

        return loss_contrastive

def fit_epoch(net, trainloader, writer, epoch=1):
    net.train()
    optimizer = optim.Adam(net.parameters())
    criterion = ContrastiveLoss()
    loss = 0.0

    for i, data in enumerate(tqdm(trainloader)):

        # get the inputs; data is a list of [inputs, labels]
        inputs1, inputs2, loss_mul, labels = data['fist_text'].to('cuda'), data['second_text'].to('cuda'), data['loss_mul'].to('cuda'), data['label'].to('cuda')

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs1, outputs2 = net(inputs1, inputs2)
        loss = criterion(outputs1, outputs2, labels)
        loss = torch.dot(loss,loss_mul)
        loss.backward()
        optimizer.step()

        loss += loss.item()

    # ...log the running loss
    writer.add_scalar('training loss', loss/len(trainloader), epoch)

def fit(net, trainloader, validationloader, epochs=1000):
    log_datatime = str(datetime.now().time())
    writer = SummaryWriter(os.path.join('logs', log_datatime))
    best_acc = 0
    for epoch in range(epochs):
        fit_epoch(net, trainloader, writer, epoch=epoch)
        train_acc = accuracy(net, trainloader, epoch)
        val_acc = accuracy(net, validationloader, epoch)
        writer.add_scalars('Accuracy', {'train':train_acc, 'validation':val_acc}, epoch)
        if val_acc>best_acc:
            best_acc = val_acc
            print('Saving model with accuracy: {}'.format(val_acc))
            torch.save(net, os.path.join('checkpoints', 'checkpoints.pth'))
        else:
            print('Epoch {} accuracy: {}'.format(epoch, val_acc))
    print('Finished Training')