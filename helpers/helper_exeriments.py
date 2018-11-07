import datapreparation
import pandas as pd

# Filpath relative to root folder
fs1_rolls = 'datasets/training/piano_roll_fs1/'

# First just try and read all the csvs
all_data = datapreparation.load_all_dataset(fs1_rolls)
print(all_data[0])
