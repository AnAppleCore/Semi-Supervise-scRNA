import argparse

parser = argparse.ArgumentParser()
# choose the GPU id (0,1,2,3)
parser.add_argument('-d', '--device_index', type = str, default = '1')
parser.add_argument('-p', '--prior_var', type = float, default = 1.)
parser.add_argument('-s', '--samples', type = int, default = 10)
parser.add_argument('-r', '--learning_rate', type = float, default = 1e-3)
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


class BayesLinear(nn.Module):
    """
        Bayes Linear layer
        Using backpropagation.
    """
    def __init__(self, input_features, output_features, prior_var=1.):
        """
            Initialization of BNN layer
            
            prior is a normal distribution centered in 0 and of variance 1.
            standard normal distribution
        """
        super().__init__()

        self.input_features = input_features
        self.output_features = output_features

        # initialize mu and rho parameters for the weights of the layer
        self.w_mu = nn.Parameter(torch.zeros(output_features, input_features))
        self.w_rho = nn.Parameter(torch.zeros(output_features, input_features))

        #initialize mu and rho parameters for the layer's bias
        self.b_mu =  nn.Parameter(torch.zeros(output_features))
        self.b_rho = nn.Parameter(torch.zeros(output_features))        

        #initialize weight samples, calculated whenever the layer makes a prediction
        self.w = None
        self.b = None

        # initialize prior distribution for all of the weights and biases
        self.prior = torch.distributions.Normal(0,prior_var)

    def forward(self, input):
        """
          Optimization process
        """
        # sample weights
        w_epsilon = Normal(0,1).sample(self.w_mu.shape).cuda()
        self.w = self.w_mu + torch.log(1+torch.exp(self.w_rho)) * w_epsilon

        # sample bias
        b_epsilon = Normal(0,1).sample(self.b_mu.shape).cuda()
        self.b = self.b_mu + torch.log(1+torch.exp(self.b_rho)) * b_epsilon

        # record log prior by evaluating log pdf of prior at sampled weight and bias
        w_log_prior = self.prior.log_prob(self.w)
        b_log_prior = self.prior.log_prob(self.b)
        self.log_prior = torch.sum(w_log_prior) + torch.sum(b_log_prior)

        # record log variational posterior by evaluating log pdf of normal distribution defined by parameters with respect at the sampled values
        self.w_post = Normal(self.w_mu.data, torch.log(1+torch.exp(self.w_rho)))
        self.b_post = Normal(self.b_mu.data, torch.log(1+torch.exp(self.b_rho)))
        self.log_post = self.w_post.log_prob(self.w).sum() + self.b_post.log_prob(self.b).sum()

        return F.linear(input, self.w, self.b)


class BNN(nn.Module):
    def __init__(self, input_features, hidden_units, output_features, noise_tol=.1,  prior_var=1.):

        # initialize the network like you would with a standard multilayer perceptron, but using the BBB layer
        super().__init__()
        self.hidden = BayesLinear(input_features,hidden_units, prior_var=prior_var)
        self.out = BayesLinear(hidden_units, output_features, prior_var=prior_var)
        self.noise_tol = noise_tol # we will use the noise tolerance to calculate our likelihood
        self.input_features = input_features
        self.output_features = output_features

    def forward(self, x):
        # again, this is equivalent to a standard multilayer perceptron
        x = torch.sigmoid(self.hidden(x))
        x = self.out(x)
        return x

    def log_prior(self):
        # calculate the log prior over all the layers
        return self.hidden.log_prior + self.out.log_prior

    def log_post(self):
        # calculate the log posterior over all the layers
        return self.hidden.log_post + self.out.log_post

    def sample_elbo(self, input, target, samples = 1):
        # we calculate the negative elbo, which will be our loss function
        #initialize tensors
        # samples is the number of "predictions" we make for 1 x-value.
        outputs = torch.zeros(samples, target.shape[0], target.shape[1]).cuda()
        log_priors = torch.zeros(samples)
        log_posts = torch.zeros(samples)
        log_likes = torch.zeros(samples)
        # make predictions and calculate prior, posterior, and likelihood for a given number of samples
        for i in range(samples):
            outputs[i] = self(input) # make predictions
            log_priors[i] = self.log_prior() # get log prior
            log_posts[i] = self.log_post() # get log variational posterior
            log_likes[i] = Normal(outputs[i], self.noise_tol).log_prob(target).sum() # calculate the log likelihood
        # calculate monte carlo estimate of prior posterior and likelihood
        log_prior = log_priors.mean()
        log_post = log_posts.mean()
        log_like = log_likes.mean()
        # calculate the negative elbo (which is our loss function)
        loss = log_post - log_prior - log_like
        return loss


def run_BNN(DataPath, LabelsPath, CV_RDataPath, OutputDir, GeneOrderPath = "", NumGenes = 0):
    '''
    run baseline classifier: BNN
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
    print('Classes:', unique)
    Y = np.zeros([len(labels), len(unique)], int)
    for j in range(len(unique)):
        Y[np.where(labels == unique[j])[0], j] = 1
    Y = torch.Tensor(Y)
    # print(Y.shape)
    # print(len(Y[0]))

    # set the classifier
    Classifier = BNN(len(X[0]), 1024, len(Y[0]), prior_var=opt.prior_var) .cuda()
    optimizer = optim.Adam(Classifier.parameters(), lr=opt.learning_rate)
            
    tr_time=[]
    ts_time=[]
    true_lab = np.zeros([len(labels), 1], dtype = int)
    pred_lab = np.zeros([len(labels), 1], dtype = int)
    

    for i in range(np.squeeze(nfolds)):
        test_ind_i = np.array(test_ind[i], dtype = 'int') - 1
        train_ind_i = np.array(train_ind[i], dtype = 'int') - 1

        x_train = X[train_ind_i].cuda()
        x_test = X[test_ind_i].cuda()
        y_train = Y[train_ind_i].cuda()
        y_test = Y[test_ind_i]
            
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
        epochs = 100
        for epoch in range(epochs):
            loss = Classifier.sample_elbo(x_train, y_train, opt.samples)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (epoch+1) % 10 == 0:
                print('\tepoch: {}/{}'.format(epoch+1, epochs))
                print('\tLoss: ', loss.item(), '\n')
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
        truelab.to_csv("BNN_True_Labels.csv", index = False)
        predlab.to_csv("BNN_Pred_Labels.csv", index = False)
        tr_time.to_csv("BNN_Training_Time.csv", index = False)
        ts_time.to_csv("BNN_Testing_Time.csv", index = False)
    else:
        truelab.to_csv("BNN_" + str(NumGenes) + "_True_Labels.csv", index = False)
        predlab.to_csv("BNN_" + str(NumGenes) + "_Pred_Labels.csv", index = False)
        tr_time.to_csv("BNN_" + str(NumGenes) + "_Training_Time.csv", index = False)
        ts_time.to_csv("BNN_" + str(NumGenes) + "_Testing_Time.csv", index = False)

if torch.cuda.is_available():
    print('The code uses GPU ', opt.device_index)
    
run_BNN('../scRNA_datasets/Pancreatic_data/Baron Human/Filtered_Baron_HumanPancreas_data.csv','../scRNA_datasets/Pancreatic_data/Baron Human/Labels.csv','../scRNA_datasets/Pancreatic_data/Baron Human/CV_folds.RData','./Results/Pancreatic_data/Baron Human/BNN/')
