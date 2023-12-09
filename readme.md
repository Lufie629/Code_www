# Dirs structure 
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
        |--Data
        |--process_code
        |--process_data
    |--Proposed_method  
        |--Proposed_method.log
        |--Proposed_method.py
        |--tweet_layer.py
        |--proposed_method.sh
    |--readme.md
    |--requirements.txt  

```
## Init
```
Before testing the code, it's necessary to prepare the environment in which the code will run.
Therefore, please first execute the 'Init.sh' shell file.
```
```
# Init.sh
# step1: prepare the enviroment
pip install -r requriments.txt

# step2: get the datasets, and you can alse download them from the url:https://drive.google.com/file/d/1T_WzrscrP1tzDVBDYFv6K7HAVT9FkJ7x/view?usp=drive_link
gdown https://drive.google.com/uc?id=1T_WzrscrP1tzDVBDYFv6K7HAVT9FkJ7x

# step3: The compressed file name is 'Datasets_process.tar.gz'. After decompression, you will obtain the 'Datasets_process' folder. The directory structure is as shown above.
tar -xzvf Datasets_process.tar.gz
```
## Datasets_process
```
Regarding the data processing part, we have placed all the data, process files, and data processing code used throughout the experimental process in the 'Datasets_process' folder. 
This folder contains our approach and logic for data processing. 
If you wish to gain a thorough understanding of these, please patiently review the files in the code section.
```

## Baseline
```
For all the baseline methods in the paper, their code is placed in the 'baseline' folder. 
If you wish to run them (for training or testing), you can execute the desired method according to the 'bashline.sh'.

please 'bash baseline.sh'
or read baseline.sh
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



