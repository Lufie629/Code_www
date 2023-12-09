# Init.sh
# step1: prepare the enviroment
pip install -r requriments.txt

# step2: get the datasets, and you can alse download them from the url:https://drive.google.com/file/d/1_NFdFUKpXdLbcsb1e49fNAgKG9EzyT4a/view?usp=drive_link
gdown https://drive.google.com/uc?id=1_NFdFUKpXdLbcsb1e49fNAgKG9EzyT4a

# step3: The compressed file name is 'Datasets_process.tar.gz'. After decompression, you will obtain the 'Datasets_process' folder. The directory structure is as shown above.
tar -xzvf Datasets_process.tar.gz
