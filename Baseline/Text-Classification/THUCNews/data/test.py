import numpy as np

# Load the .npz file
data = np.load('/data/gluo/Chinese-Text-Classification-Pytorch-master/THUCNews/data/embedding_SougouNews.npz')

# Access the arrays saved in the file
array1 = data['array1']
array2 = data['array2']

print(data)

# You can now use the loaded arrays in your code
