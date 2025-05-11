import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn
from models.gat_gcn import GAT_GCNNet
from models.gin_gcn import GIN_GCNNet
from utils import *

# training function at each epoch
def train(model, device, train_loader_drug,train_loader_protein, optimizer, epoch):
    print('Training on {} samples...'.format(len(train_loader_drug.dataset)))
    model.train()
    for batch_idx, (data_drug,data_protein) in enumerate(zip(train_loader_drug,train_loader_protein)):
        data_drug = data_drug.to(device)
        data_protein = data_protein.to(device)
        optimizer.zero_grad()
        output = model(data_drug,data_protein)
        loss = loss_fn(output, data_drug.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data_drug.x),
                                                                           len(train_loader_drug.dataset),
                                                                           100. * batch_idx / len(train_loader_drug),
                                                                           loss.item()))

def predicting(model, device, loader_drug,loader_protein):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader_drug.dataset)))
    with torch.no_grad():
        for data_drug,data_protein in zip(loader_drug,loader_protein):
            data_drug = data_drug.to(device)
            data_protein = data_protein.to(device)
            output = model(data_drug,data_protein)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data_drug.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(),total_preds.numpy().flatten()


datasets = [['davis','kiba'][int(sys.argv[1])]] 
modeling = [GIN_GCNNet, GAT_GCNNet][int(sys.argv[2])]
model_st = modeling.__name__

cuda_name = "cuda:0"
if len(sys.argv)>3:
    cuda_name = "cuda:" + str(int(sys.argv[3])) 
print('cuda_name:', cuda_name)

TRAIN_BATCH_SIZE = 256
TEST_BATCH_SIZE = 256
LR = 0.0005
LOG_INTERVAL = 20
NUM_EPOCHS = 500

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

# Main program: iterate over different datasets
for dataset in datasets:
    print('\nrunning on ', model_st + '_' + dataset )
    processed_data_file_train_drug = 'data/processed/' + dataset + '_train_drug.pt'
    processed_data_file_test_drug = 'data/processed/' + dataset + '_test_drug.pt'
    processed_data_file_train_protein = 'data/processed/' + dataset + '_train_protein.pt'
    processed_data_file_test_protein = 'data/processed/' + dataset + '_test_protein.pt'
    if ((not os.path.isfile(processed_data_file_train_drug)) or (not os.path.isfile(processed_data_file_test_drug)) or (not os.path.isfile(processed_data_file_train_protein)) or (not os.path.isfile(processed_data_file_test_protein))):
        print('please run create_data.py to prepare data in pytorch format!')
    else:
        train_data_drug = TestbedDataset(root='data', dataset=dataset+'_train_drug')
        test_data_drug = TestbedDataset(root='data', dataset=dataset+'_test_drug')
        train_data_protein = TestbedDataset(root='data', dataset=dataset+'_train_protein')
        test_data_protein = TestbedDataset(root='data', dataset=dataset+'_test_protein')
        
        # make data PyTorch mini-batch processing ready
        train_loader_drug = DataLoader(train_data_drug, batch_size=TRAIN_BATCH_SIZE, shuffle=False)
        test_loader_drug = DataLoader(test_data_drug, batch_size=TEST_BATCH_SIZE, shuffle=False)
        train_loader_protein = DataLoader(train_data_protein, batch_size=TRAIN_BATCH_SIZE, shuffle=False)
        test_loader_protein = DataLoader(test_data_protein, batch_size=TEST_BATCH_SIZE, shuffle=False)

        # training the model
        device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
        model = modeling().to(device)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        best_mse = 1000
        best_ci = 0
        best_epoch = -1
        model_file_name = 'model_' + model_st + '_' + dataset +  '.model'
        result_file_name = 'result_' + model_st + '_' + dataset +  '.csv'
        for epoch in range(NUM_EPOCHS):
            train(model, device, train_loader_drug,train_loader_protein, optimizer, epoch+1)
            G,P = predicting(model, device, test_loader_drug,test_loader_protein)
            ret = [rmse(G,P),mse(G,P),pearson(G,P),spearman(G,P),ci(G,P)]
            if ret[1]<best_mse:
                torch.save(model.state_dict(), model_file_name)
                with open(result_file_name,'w') as f:
                    f.write(','.join(map(str,ret)))
                best_epoch = epoch+1
                best_mse = ret[1]
                best_ci = ret[-1]
                print('rmse improved at epoch ', best_epoch, '; best_mse,best_ci:', best_mse,best_ci,model_st,dataset)
            else:
                print(ret[1],'No improvement since epoch ', best_epoch, '; best_mse,best_ci:', best_mse,best_ci,model_st,dataset)

