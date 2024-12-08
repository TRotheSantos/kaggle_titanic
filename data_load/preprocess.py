import numpy as np
import pandas as pd
import ethnicolr
import re
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)

files = []

# for dirname, _, filenames in os.walk('../data'):
#     for filename in filenames:
#
#         files.append(os.path.join(dirname, filename))
#
# df = pd.read_csv(files[0])  # train set
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


def split_name(name):
    # Ensure the input is a string before processing
    if isinstance(name, str):
        match = re.match(r'([^,]+),\s*(?:[A-Za-z]+\.\s*)?([A-Za-z]+)', name)
        if match:
            last_name = match.group(1)
            first_name = match.group(2)  # Get the first part of the first name
            return pd.Series([last_name, first_name])
    return pd.Series([None, None])


scaler = MinMaxScaler()

def probable_ethnicity(df):
    # CULTURAL ENCODING for Names using ethnicolr
    # Split the 'Name' column into first and last names
    df[['last_name', 'first_name']] = df['Name'].apply(split_name)

    df = ethnicolr.pred_wiki_name(df, 'last_name', 'first_name', conf_int=0.9)

    print(df['race'].head)

    return df['race']

def encode_training_data(path):

    df = pd.read_csv(path)

    df = df.drop(columns=["PassengerId"])   # remove index

    label = df.pop("Survived")

    df['Pclass'] = scaler.fit_transform(df[['Pclass']])

    # NAMES
    df['ethnicity'] = probable_ethnicity(df)

    # EMBARKED
    df = pd.get_dummies(df, columns=["Embarked"])

    # TICKETS
    df['Ticket_Letters'] = df['Ticket'].apply(lambda x: re.sub(r'[^a-zA-Z/]', '', x))
    df['Ticket_Letters'] = df['Ticket_Letters'].str[0]
    df = pd.get_dummies(df, columns=["Ticket_Letters"])
    df['Ticket_Digit_Length'] = df['Ticket'].apply(lambda x: len(re.sub(r'\D', '', x)))
    df = df.drop(columns=["Ticket"])

    # CABIN
    df['Cabin'] = df['Cabin'].str[0]
    df['Cabin_Scaled'] = df['Cabin'].map(deck_mapping).fillna(0)
    df = df.drop(columns=['Cabin'])

    # print(df.head(n=10))

    x = df
    y = label

    return x, y
