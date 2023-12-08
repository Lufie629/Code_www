# current dir is in Code-------------------------------------------------------
pip install -r requirements.txt
# -------------------------------------------------------

# test Miller
cd Miller
python stream_cluster.py
cd ../
# -------------------------------------------------------

# test Kudugunta
cd Kudugunta
python train.py
cd ../
# -------------------------------------------------------

# test Wei
cd Wei
python bilstm_attention.py
cd ../
# -------------------------------------------------------

# test Lee
cd Lee
python train.py
cd ../
# -------------------------------------------------------


# test TextRNN, TextRNN-Att, FastText, DPCNN, TextRNN, TextCNN, please choose the model name!
cd Text-Classification
python run.py
cd ../
# -------------------------------------------------------


# test Alhosseini
cd Alhosseini
python gcntwi22.py
cd ../
# -------------------------------------------------------


# test BotRGCN
cd BotRGCN
python main.py
cd ../
# -------------------------------------------------------

# test RGT
cd RGT
python RGT.py
cd ../
# -------------------------------------------------------