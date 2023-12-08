# dirs structure 
```
|--Code  
    |--Baseline  
        |--datasets  
        |--Alhosseini  
        |--BotRGCN  
        |--Kudugunta  
        |--Lee  
        |--Miller  
        |--Text-Classification    
        |--Wei  
        |--baseline.sh
    |--Datasets_process  
    |--Proposed_method  
        |--Proposed_method.log
        |--Proposed_method.py
        |--tweet_layer.py
        |--proposed_method.sh
    |--readme.md
    |--requirements.txt  

```


## Baseline
```
For all the baseline methods in the paper, their code is placed in the 'baseline' folder. 
If you wish to run them (for training or testing), you can execute the desired method according to the 'bashline.sh'.

please 'bash baseline.sh'
or read baseline.sh
```

## Datasets_process
```
Regarding the data processing part, we have placed all the data, process files, and data processing code used throughout the experimental process in the 'Datasets_process' folder. 
This folder contains our approach and logic for data processing. 
If you wish to gain a thorough understanding of these, please patiently review the files in the code section.
```

## Proposed method
```
The implementation code of the method proposed in this article is placed in the specified folder, which contains three files: 'Proposed_method.py', 'tweet_layer.py', and 'Proposed_method.log'. 
Among them, 'Proposed_method.log' is a log file recording experimental data. 
If you want to train this model, you can directly run the 'Proposed_method.py' file. 
We have already organized the data paths. 
If there are any issues, please flexibly refer to the code. 
The datasets are all placed in the 'Datasets_process' folder.

please 'bash proposed_method.sh'
or read proposed_method.sh
```



