import numpy as np
import pandas as pd
import os
import torch.nn as nn
import ethnicolr
import re
pd.set_option('display.max_columns', None)

files = []

# for dirname, _, filenames in os.walk('../data'):
#     for filename in filenames:
#
#         files.append(os.path.join(dirname, filename))
#
# df = pd.read_csv(files[0])  # train set

df = pd.read_csv('data/train.csv')


# Prepare the data

df = df.drop(columns=["PassengerId"])
label = df.pop("Survived")


df = pd.get_dummies(df, columns=["Embarked"])

df['Ticket_Letters'] = df['Ticket'].apply(lambda x: re.sub(r'[^a-zA-Z/]', '', x))
df['Ticket_Letters'] = df['Ticket_Letters'].str[0]
df = pd.get_dummies(df, columns=["Ticket_Letters"])
df['Ticket_Digit_Length'] = df['Ticket'].apply(lambda x: len(re.sub(r'\D', '', x)))
df = df.drop(columns=["Ticket"])


deck_mapping = {
    'A': 1,
    'B': 0.9,
    'C': 0.8,
    'D': 0.7,
    'E': 0.6,
    'F': 0.5,
    'G': 0.4,
    'T': 0.3  # Unusual case, only 1 passenger in the Titanic dataset
}
df['Cabin'] = df['Cabin'].str[0]
df['Cabin_Scaled'] = df['Cabin'].map(deck_mapping).fillna(0)
df = df.drop(columns=['Cabin'])



print(df.head(n=10))


class simpleMLP(nn.Module):
    def __init__(self, input_size):
        super(simpleMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.simpleMLP(x)




