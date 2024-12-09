from data_load import data_handle
from model import simple_MLP


train_file = 'data/train.csv'
test_file = 'data/test.csv'

X_train, X_test, y_train = data_handle.data_loader(train_file, test_file, save=True)



