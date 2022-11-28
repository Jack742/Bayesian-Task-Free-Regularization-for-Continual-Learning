import os
import pickle 

print(os.listdir('Results/pmnist'))

with open('Results/pmnist/mas_base', 'rb') as f:
    print(pickle.load(f))