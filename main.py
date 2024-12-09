import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam


from data_load import data_handle
# from model import simple_MLP

train_file = 'data/train.csv'
test_file = 'data/test.csv'

X_train, X_test, y_train = data_handle.data_loader(train_file, test_file, save=True)

# X_train = pd.read_csv('data/train_encoded.csv')
# X_test = pd.read_csv('data/test_encoded.csv')
# y_train = pd.read_csv('data/train_labels.csv')

print(len(X_test))

# pred_test = simple_MLP.train_balanced_mlp(X_train, X_test, y_train)

#
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
# y_train = y_train.values.ravel()
#
# class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
# class_weights_dict = dict(enumerate(class_weights))
# print("Class Weights:", class_weights_dict)
#
# model = Sequential([
#     Dense(128, input_dim=X_train.shape[1], activation='relu'),  # Input-Schicht
#     BatchNormalization(),
#     Dropout(0.3),
#
#     Dense(64, activation='relu'),  # Verborgene Schicht
#     BatchNormalization(),
#     Dropout(0.3),
#
#     Dense(32, activation='relu'),  # Weitere verborgene Schicht
#     BatchNormalization(),
#     Dropout(0.3),
#
#     Dense(1, activation='sigmoid')  # Ausgangsschicht (Binary Classification)
# ])
#
# # Modell kompilieren
# model.compile(optimizer=Adam(learning_rate=0.001),
#               loss='binary_crossentropy',
#               metrics=['accuracy'])
#
#
# history = model.fit(X_train, y_train,
#                     epochs=50,
#                     batch_size=32,
#                     validation_split=0.2,
#                     class_weight=class_weights_dict,  # Klassengewichtung
#                     verbose=1)
#
# loss, accuracy = model.evaluate(X_train, y_train)
# print(f"Train Accuracy: {accuracy:.4f}")
#
# # Vorhersagen machen
# y_pred = model.predict(X_test)
#
# y_pred_binary = (y_pred > 0.5).astype(int)
#
# df_pred = pd.DataFrame(y_pred_binary, columns=['Survived'])
#
#
# passenger_ids = range(892, 892 + len(y_pred_binary))
# df_pred['PassengerId'] = passenger_ids
#
# df_submission = df_pred[['PassengerId', 'Survived']]
#
# print(df_submission)
# df_submission.to_csv("./predictions/submission1.csv", index=False)

