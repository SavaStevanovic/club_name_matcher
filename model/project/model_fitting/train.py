import torch.optim as optim
import torch.nn as nn
import torch
import os
from model_fitting.losses import ContrastiveLoss
from model_fitting.metrics import metrics, validation_metrics
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

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

    writer.add_scalar('training loss', loss/len(trainloader), epoch)

def fit(net, trainloader, validationloader, epochs=1000):
    log_datatime = str(datetime.now().time())
    writer = SummaryWriter(os.path.join('logs', log_datatime))
    best_acc = 0
    i = 0
    lr_rate = 0.001
    for epoch in range(epochs):
        fit_epoch(net, trainloader, writer, lr_rate, epoch=epoch)
        train_f1_score, train_acc = metrics(net, trainloader, epoch)
        val_acc = validation_metrics(net, validationloader, epoch)
        writer.add_scalars('metrics', {'val_acc':val_acc, 'train_acc':train_acc, 'train_f1_score':train_f1_score}, epoch)
        if best_acc < val_acc:
            i=0
            best_acc = val_acc
            print('Epoch {}. Saving model with acc: {}'.format(epoch, val_acc))
            chp_dir = 'checkpoints'
            os.makedirs((chp_dir), exist_ok=True)
            torch.save(net, os.path.join(chp_dir, 'checkpoints.pth'))
        else:
            i+=1
            print('Epoch {} acc: {}'.format(epoch, val_acc))
        if i==100:
            lr_rate*=0.1
            i=0
            print("Learning rate lowered to {}".format(lr_rate))
    print('Finished Training')