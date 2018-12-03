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

### The overall performance
<table>
        <tr>
            <th> </th>
            <th>TID13-LCC</th>
            <th>TID13-SROCC</th>
            <th>LIVE-LCC</th>
            <th>LIVE-SROCC</th>
        </tr>
        <tr>
            <th>deep QA</th>
            <th>0.947</th>
            <th>0.939</th>
            <th>0.982</th>
            <th>0.981</th>
        </tr>
        <tr>
            <th>This code</th>
            <th>0.929</th>
            <th>0.924</th>
            <th>0.971</th>
            <th>0.967</th>
        </tr>
    </table>

The "results" folder contains the training process and two examples for both LIVE and TID2013 datasets.

### The training process:

**LIVE** 
<div align=center><img width="600" src="https://github.com/LeonLIU08/DeepQA-with-Pytorch/blob/master/results/LIVEtrainhist.png?raw=true"/></div>

**TID2013**
<div align=center><img width="600" src="https://github.com/LeonLIU08/DeepQA-with-Pytorch/blob/master/results/TID2013trainhist.png?raw=true"/></div>

