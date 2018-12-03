# DeepQA-with-Pytorch

This project aims to reimplement the work "Deep Learning of Human Visual Sensitivity in Image Quality Assessment Framework" for FR-IQA on PyTorch platform. The code has been trained and tested on LIVE and TID2013 database. The current performance is close to the claimed performance in the original paper. 

## File structure

This project folder should be included in a upper directory with the database data as following:

```
project
│   README.md
│   file001.txt    
│
└───folder1
│   │   file011.txt
│   │   file012.txt
│   │
│   └───subfolder1
│       │   file111.txt
│       │   file112.txt
│       │   ...
│   
└───folder2
    │   file021.txt
    │   file022.txt
```

`--mother folder`

`----data`

'------LIVE_dataset'

'------TID2013_dataset'

'----this code'
