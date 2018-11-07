import datapreparation as dp
from dataset import pianoroll_dataset_batch

# Filpath relative to root folder
fs1_rolls = 'datasets/training/piano_roll_fs1/'

# First just try and read all the csvs
all_data = dp.load_all_dataset(fs1_rolls)
print(all_data[0])
print(all_data[0].shape)
print('Num songs:', len(all_data))

# Try and read the pytorch dataset
dataset_batch = pianoroll_dataset_batch(fs1_rolls) 