import pandas as pd

from data_load import preprocess


train_file = 'data/train.csv'

x, y = preprocess.encode_training_data(train_file)








