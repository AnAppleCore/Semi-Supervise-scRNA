import argparse

parser = argparse.ArgumentParser()
# choose the GPU id (0,1,2,3)
parser.add_argument('-d', '--device_index', type = str, default = '1')
opt = parser.parse_args()

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = opt.device_index

import numpy as np
import pandas as pd
import time as tm
from scipy.stats import norm
import rpy2.robjects as robjects

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal


class FC(nn.Module):
    def __init__(self, input_features, hidden_units, output_features, noise_tol=.1,  prior_var=1.):

        super().__init__()
        self.hidden = nn.Linear(input_features,hidden_units)
        self.out = nn.Linear(hidden_units, output_features)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.out(x)
        return x


def run_FC(DataPath, LabelsPath, CV_RDataPath, OutputDir, GeneOrderPath = "", NumGenes = 0):
    '''
    run baseline classifier: FC
    Train on the target dataset
    '''
        
    # read the Rdata file
    robjects.r['load'](CV_RDataPath)

    nfolds = np.array(robjects.r['n_folds'], dtype = 'int')
    tokeep = np.array(robjects.r['Cells_to_Keep'], dtype = 'bool')
    col = np.array(robjects.r['col_Index'], dtype = 'int')
    col = col - 1 
    test_ind = np.array(robjects.r['Test_Idx'], dtype=object)
    train_ind = np.array(robjects.r['Train_Idx'], dtype=object)

    # read the data
    data = pd.read_csv(DataPath,index_col=0,sep=',')
    labels = pd.read_csv(LabelsPath, header=0,index_col=None, sep=',', usecols = col)
    
    labels = labels.iloc[tokeep]
    data = data.iloc[tokeep]
    print('Labels shape:', labels.shape)
    print('Data shape:', data.shape)
    
    # read the feature file
    if (NumGenes > 0):
        features = pd.read_csv(GeneOrderPath,header=0,index_col=None, sep=',')
    
    # folder with results
    os.chdir(OutputDir)
    
    # normalize data
    data = np.log1p(data)
    X = np.array(data)
    X = torch.Tensor(X)
    # print(X.shape)
    # print(len(X[0]))

    # preprocess the data
    unique = np.unique(labels)
    print('Classes: ', unique)
    Y = np.zeros([len(labels), len(unique)], int)
    for j in range(len(unique)):
        Y[np.where(labels == unique[j])[0], j] = 1
    Y = torch.LongTensor(Y)
    # print(Y.shape)
    # print(len(Y[0]))

    # set the classifier
    Classifier = FC(len(X[0]), 1024, len(Y[0])).cuda()
    optimizer = optim.Adam(Classifier.parameters(), lr=1e-3)
            
    tr_time=[]
    ts_time=[]
    true_lab = np.zeros([len(labels), 1], dtype = int)
    pred_lab = np.zeros([len(labels), 1], dtype = int)
    

    for i in range(np.squeeze(nfolds)):
        test_ind_i = np.array(test_ind[i], dtype = 'int') - 1
        train_ind_i = np.array(train_ind[i], dtype = 'int') - 1

        x_train = X[train_ind_i].cuda()
        x_test = X[test_ind_i].cuda()
        y_train = Y[train_ind_i]
        y_test = Y[test_ind_i]

        train_lab = np.zeros([len(train_ind_i)], dtype = int)
        for j in range(len(train_ind_i)):
            train_lab[j] = np.argmax(y_train[j], axis = 0)
        train_lab = torch.LongTensor(train_lab).cuda()
            
        if (NumGenes > 0):
            feat_to_use = features.iloc[0:NumGenes,i]
            x_train = x_train[:,feat_to_use]
            x_test = x_test[:,feat_to_use]
        
        print('Fold:', i)
        print('\tTest #:', len(test_ind_i))
        print('\tTrain #:', len(train_ind_i), '\n')
        # print('\tx_train.shape: ', x_train.shape)
        # print('\tx_test.shape: ', x_test.shape)
        # print('\ty_train.shape: ', y_train.shape)
        # print('\ty_test.shape: ', y_test.shape, '\n')


        start=tm.time()
        epochs = 10
        for epoch in range(epochs):
            output = Classifier(x_train)
            loss = F.cross_entropy(output, train_lab)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (epoch+1) % 10 == 0:
                print('\tepoch: {}/{}'.format(epoch+1, epochs))
                print('\tLoss: %.3f\n'% loss.item())
        tr_time.append(tm.time()-start)
                    
        start=tm.time()
        y_pred = Classifier(x_test).detach().cpu().numpy()
        # print('\ty_pred.shape: ', y_pred.shape)
        ts_time.append(tm.time()-start)
    
        for i in range(len(test_ind_i)):
            true_lab[test_ind_i[i], 0] = np.argmax(y_test[i], axis = 0)
            pred_lab[test_ind_i[i], 0] = np.argmax(y_pred[i], axis = 0)

    truelab = [unique[true_lab[i, 0]] for i in range(len(labels))]
    predlab = [unique[pred_lab[i, 0]] for i in range(len(labels))]
                
    truelab = pd.DataFrame(truelab)
    predlab = pd.DataFrame(predlab)
        
    tr_time = pd.DataFrame(tr_time)
    ts_time = pd.DataFrame(ts_time)
        
    if (NumGenes == 0):  
        truelab.to_csv("FC_True_Labels.csv", index = False)
        predlab.to_csv("FC_Pred_Labels.csv", index = False)
        tr_time.to_csv("FC_Training_Time.csv", index = False)
        ts_time.to_csv("FC_Testing_Time.csv", index = False)
    else:
        truelab.to_csv("FC_" + str(NumGenes) + "_True_Labels.csv", index = False)
        predlab.to_csv("FC_" + str(NumGenes) + "_Pred_Labels.csv", index = False)
        tr_time.to_csv("FC_" + str(NumGenes) + "_Training_Time.csv", index = False)
        ts_time.to_csv("FC_" + str(NumGenes) + "_Testing_Time.csv", index = False)

if torch.cuda.is_available():
    print('The code uses GPU ', opt.device_index)

run_FC('../scRNA_datasets/Pancreatic_data/Baron Human/Filtered_Baron_HumanPancreas_data.csv','../scRNA_datasets/Pancreatic_data/Baron Human/Labels.csv','../scRNA_datasets/Pancreatic_data/Baron Human/CV_folds.RData','./Results/Pancreatic_data/Baron Human/FC/')
