import torch
from model.LSTM import LSTMNet
from model_fitting.train import fit
from data_loader.dataset_creator import DatasetCreator

th_count = 12

dataset_creator = DatasetCreator(root_dir = './dataset', names_file = 'data_loader/universalnames.json')

for i in range(6):
    for j in range(5):
        width = 16*2**i
        depth = 1+j
        net = LSTMNet(len(dataset_creator.corpus), width, depth)
        net.cuda()

        trainset = dataset_creator.get_train_iterator()
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=th_count, pin_memory=True)

        validationset = dataset_creator.get_validation_iterator()
        validationloader = torch.utils.data.DataLoader(validationset, batch_size=1, shuffle=False, num_workers=th_count, pin_memory=False)

        fit(net, trainloader, validationloader, chp_prefix="{}_{}".format(width, depth), epochs=100, lower_learning_period=10)