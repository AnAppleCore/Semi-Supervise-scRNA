# A comparison of automatic cell identification methods for single-cell RNA-sequencing data

## TO DO:

Implement the scVI, BNN, scVI+FC and FC model


### General Usage

To benchmark and fairly evaluate the performance of different classifiers using benchmark-datasets (Filtered datasets can be downloaded from https://zenodo.org/record/3357167), apply the following steps:

#### Step 1

Apply the ```Cross_Validation``` R function on the corresponding dataset to obtain fixed training and test cell indices, straitified across different cell types. For example, using the Tabula Muris (TM) dataset

```R
Cross_Validation('~/TM/Labels.csv', 1, '~/TM/')
```

This command will create a ```CV_folds.RData``` file used as input in Step 2.

#### Step 2

Run each classifier wrapper. For example, running scPred on TM dataset

```R
run_scPred('~/TM/Filtered_TM_data.csv','~/TM/Labels.csv','~/TM/CV_folds.RData','~/Results/TM/')
```

This command will output the true and predicted cell labels as csv files, as well as the classifier computation time.

#### Step 3

Evaluate the classifier prediction by 

```R
result <- evaluate('~/Results/TM/scPred_True_Labels.csv', '~/Results/TM/scPred_Pred_Labels.csv')
```

This command will return the corresponding accuracy, median F1-score, F1-scores for all cell populations, % unlabeled cells, and confusion matrix.
