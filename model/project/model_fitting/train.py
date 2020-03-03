import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score

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

def metrics( net, dataloader, epoch=1):
    net.eval()
    correct = 0
    total = 0
    predictions = []
    labels = []
    with torch.no_grad():
        for data in dataloader:
            inputs1, inputs2, loss_mul, label = data['first_text'].to('cuda'), data['second_text'].to('cuda'), data['loss_mul'].to('cuda'), data['label'].to('cuda')
            outputs1, outputs2 = net(inputs1, inputs2)
            loss = (outputs2 - outputs1).pow(2).sum(1)
            predicted = loss < 0.5
            predictions+=predicted.cpu().numpy().tolist()
            labels+=label.cpu().numpy().tolist()
    print("Class balance {}".format(sum(labels)/len(labels)))
    return f1_score(labels, predictions), accuracy_score(labels, predictions)

def fit_epoch(net, trainloader, writer, lr_rate, epoch=1):
    net.train()
    optimizer = optim.Adam(net.parameters(), lr_rate)
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
    best_f1 = 0
    i = 0
    lr_rate = 0.001
    for epoch in range(epochs):
        fit_epoch(net, trainloader, writer, lr_rate, epoch=epoch)
        val_loss = validate_epoch(net, validationloader, writer, epoch)
        train_f1_score, train_acc = metrics(net, trainloader, epoch)
        val_f1_score, val_acc = metrics(net, validationloader, epoch)
        writer.add_scalars('metrics', {'val_acc':val_acc, 'val_f1_score':val_f1_score, 'train_acc':train_acc, 'train_f1_score':train_f1_score}, epoch)
        if best_f1 < val_f1_score:
            i=0
            best_f1 = val_f1_score
            print('Epoch {}. Saving model with acc: {}, and f1 score: {}'.format(epoch, val_acc, val_f1_score))
            chp_dir = 'checkpoints'
            os.makedirs((chp_dir), exist_ok=True)
            torch.save(net, os.path.join(chp_dir, 'checkpoints.pth'))
        else:
            i+=1
            print('Epoch {} acc: {}, and f1 score: {}'.format(epoch, val_acc, val_f1_score))
        if i==100:
            lr_rate*=0.1
            i=0
            print("Learning rate lowered to {}".format(lr_rate))
    print('Finished Training')