import torch
from model.LSTM import LSTMNet
from model_fitting.train import fit
from data_loader.dataset_creator import DatasetCreator

th_count = 12

dataset_creator = DatasetCreator(root_dir = './dataset', names_file = 'data_loader/universalnames.json')

net = LSTMNet(len(dataset_creator.corpus), 5)
net.cuda()


trainset = dataset_creator.get_train_iterator()
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=th_count, pin_memory=True)

validationset = dataset_creator.get_validation_iterator()
validationloader = torch.utils.data.DataLoader(validationset, batch_size=256, shuffle=False, num_workers=th_count, pin_memory=True)

fit(net, trainloader, validationloader, epochs=100)