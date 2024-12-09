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

# mapping in order of closeness to Britain
ethnicity_mapping = {
    "GreaterEuropean,British": 1.0,
    "GreaterEuropean,WestEuropean,Germanic": 0.9167,
    "GreaterEuropean,WestEuropean,French": 0.8333,
    "GreaterEuropean,WestEuropean,Nordic": 0.7500,
    "GreaterEuropean,WestEuropean,Italian": 0.6667,
    "GreaterEuropean,WestEuropean,Hispanic": 0.5833,
    "GreaterEuropean,EastEuropean": 0.5000,
    "GreaterEuropean,Jewish": 0.4167,
    "GreaterAfrican,Africans": 0.3333,
    "GreaterAfrican,Muslim": 0.2500,
    "Asian,IndianSubContinent": 0.1667,
    "Asian,GreaterEastAsian,Japanese": 0.0833,
    "Asian,GreaterEastAsian,EastAsian": 0.0
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

# Function to extract ticket prefix
def extract_ticket_prefix(ticket):
    # Match alphanumeric prefixes (e.g., A/5, PC)
    match = re.match(r"([A-Za-z]+[\/]?\d*)", ticket)
    return match.group(0) if match else 'No_Prefix'

# Function to extract the length of digits in the ticket
def extract_ticket_digit_length(ticket):
    # Find all digits and count their length
    digits = re.findall(r'\d', ticket)
    return len(digits)


scaler = MinMaxScaler()

def probable_ethnicity(df):
    # CULTURAL ENCODING for Names using ethnicolr
    # Split the 'Name' column into first and last names
    df[['last_name', 'first_name']] = df['Name'].apply(split_name)

    df = ethnicolr.pred_wiki_name(df, 'last_name', 'first_name', conf_int=0.9)

    df['Ethnicity_Scaled'] = df['race'].map(ethnicity_mapping).fillna(0)

    return df['Ethnicity_Scaled']


def encode_data(train_df, test_df):

    if test_df is None:
        df = train_df.copy()
    else:
        df = test_df

    df = df.drop(columns=["PassengerId"])   # remove index

    df['Pclass'] = scaler.fit_transform(df[['Pclass']])

    # NAMES
    df['Ethnicity_Scaled'] = probable_ethnicity(df)
    df = df.drop(columns=["Name"])
    df = df.drop(columns=["last_name"])
    df = df.drop(columns=["first_name"])
    df = df.drop(columns=["__name"])

    # EMBARKED
    # Calculate the survival rate for each embarkation point
    embarked_survival_rate = train_df.groupby('Embarked')['Survived'].mean()

    # Map survival rates to the 'Embarked' column
    df['Embarked_scaled'] = df['Embarked'].map(embarked_survival_rate)

    # Normalize the scale from 0 to 1 based on the minimum and maximum survival rates
    max_survival_rate = df['Embarked_scaled'].max()
    min_survival_rate = df['Embarked_scaled'].min()
    df['Embarked_scaled'] = (df['Embarked_scaled'] - min_survival_rate) / (max_survival_rate - min_survival_rate)

    # Handle NaN values if present (optional, you can assign a default value like 0)
    df['Embarked_scaled'] = df['Embarked_scaled'].fillna(df['Embarked_scaled'].mean())
    df = df.drop(columns=["Embarked"])

    # SEX
    df['sex_binary'] = df['Sex'].map({'female': 1, 'male': 0}).fillna(0)
    df = df.drop(columns=["Sex"])

    # AGE
    min_age = df['Age'].min(skipna=True)  # Minimum age, ignoring NaN
    max_age = df['Age'].max(skipna=True)  # Maximum age, ignoring NaN

    # Apply scaling formula and handle NaN
    df['age_scaled'] = (df['Age'] - min_age) / (max_age - min_age)

    df['age_scaled'] = df['age_scaled'].fillna(df['age_scaled'].mean())

    df = df.drop(columns=["Age"])

    # SIBSP / PARCH
    # Scale SibSp column
    max_sibsp = df['SibSp'].max(skipna=True)
    df['SibSp_scaled'] = df['SibSp'] / max_sibsp  # Min is already 0, so no need to subtract min

    # Scale Parch column
    max_parch = df['Parch'].max(skipna=True)
    df['Parch_scaled'] = df['Parch'] / max_parch  # Min is already 0, so no need to subtract min

    # Handle NaN values if present (optional, you can assign a default value like 0)
    df['SibSp_scaled'] = df['SibSp_scaled'].fillna(df['SibSp_scaled'].mean())
    df['Parch_scaled'] = df['Parch_scaled'].fillna(df['Parch_scaled'].mean())

    df = df.drop(columns=["SibSp"])
    df = df.drop(columns=["Parch"])

    # FARE
    max_fare = df['Fare'].max(skipna=True)
    min_fare = df['Fare'].min(skipna=True)
    df['Fare_scaled'] = (df['Fare'] - min_fare) / (max_fare - min_fare)

    # Handle NaN values if present (optional, you can assign a default value like 0 or the median)
    df['Fare_scaled'] = df['Fare_scaled'].fillna(df['Fare_scaled'].mean())
    df = df.drop(columns=["Fare"])

    # TICKETS
    df['Ticket_Prefix'] = df['Ticket'].apply(extract_ticket_prefix)
    df['Ticket_Digit_Length'] = df['Ticket'].apply(extract_ticket_digit_length)

    # Create a combined feature of Ticket Prefix and Digit Length
    train_df['Ticket_Combined'] = df['Ticket_Prefix'] + "_" + df['Ticket_Digit_Length'].astype(str)

    # Calculate survival rate for each unique combination of Ticket Prefix and Digit Length
    ticket_combined_survival_rate = train_df.groupby('Ticket_Combined')['Survived'].mean()

    # Map survival rates to the 'Ticket_Combined' column
    df['Ticket_scaled'] = train_df['Ticket_Combined'].map(ticket_combined_survival_rate)

    # Normalize the scale from 0 to 1 based on the minimum and maximum survival rates
    max_survival_rate = df['Ticket_scaled'].max()
    min_survival_rate = df['Ticket_scaled'].min()
    df['Ticket_scaled'] = (df['Ticket_scaled'] - min_survival_rate) / (max_survival_rate - min_survival_rate)

    # Handle NaN values if present (optional, you can assign a default value like 0)
    df['Ticket_scaled'] = df['Ticket_scaled'].fillna(df['Ticket_scaled'].mean())

    df = df.drop(columns=['Ticket'])
    df = df.drop(columns=['Ticket_Prefix'])
    # df = df.drop(columns=['Ticket_Combined'])
    df = df.drop(columns=['Ticket_Digit_Length'])

    # CABIN
    df['Cabin'] = df['Cabin'].str[0]
    df['Cabin_Scaled'] = df['Cabin'].map(deck_mapping)
    df['Cabin_Scaled'] = df['Cabin_Scaled'].fillna(df['Cabin_Scaled'].mean())
    df = df.drop(columns=['Cabin'])

    if test_df is None:
        label = df['Survived']
        df = df.drop(columns=['Survived'])
        df = pd.DataFrame(df.values)  # remove header
        return df, label
    else:
        df = pd.DataFrame(df.values)   # remove header
        return df


def data_loader(train_path, test_path, save=True):

    # survival_counts = df['Survived'].value_counts()
    # print(survival_counts)
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    X_train, y_train = encode_data(df_train, test_df=None)
    X_test = encode_data(df_train, df_test)

    path = './data/'
    if save:
        X_train.to_csv(path+"train_encoded.csv", index=False, header=False)
        X_test.to_csv(path+"test_encoded.csv", index=False, header=False)
        y_train.to_csv(path+"train_labels.csv", index=False, header=False)

    return X_train, X_test, y_train


def save_data(df, path):
    df.to_csv(path)





