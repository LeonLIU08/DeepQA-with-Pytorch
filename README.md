# DeepQA-with-Pytorch

This project aims to reimplement the work "Deep Learning of Human Visual Sensitivity in Image Quality Assessment Framework" for FR-IQA on PyTorch platform. The code has been trained and tested on LIVE and TID2013 database. The current performance is close to the claimed performance in the original paper. 

## File structure

This project folder should be included in a upper directory with the database data as following:

```
Project folder
└───DeepQA-with-Pytorch
│   │     README.md
│   │     train_LIVE.py
│   │     train_TID2013.py
│   └───datasets
│   └───models
│   └───snapshots
│   └───utils
│
└───data
│    └───LIVE_dataset
│    └───TID2013_dataset
```

## How to use


## Performance
The "results" folder contains the training process and two examples for both LIVE and TID2013 datasets.

The training process:

**LIVE** 
![LIVEtrain](https://github.com/LeonLIU08/DeepQA-with-Pytorch/blob/master/results/LIVEtrainhist.png?raw=true){:height="70%" width="70%"}

**TID2013**
![TID13train](https://github.com/LeonLIU08/DeepQA-with-Pytorch/blob/master/results/TID2013trainhist.png?raw=true){:height="70%" width="70%"}
