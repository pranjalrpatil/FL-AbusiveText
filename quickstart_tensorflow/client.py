import os
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding,SpatialDropout1D
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import flwr as fl
import tensorflow as tf
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
# imp--TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

max_len = 300
max_words=50000

# Load model and data (LSTM, )
model = Sequential()
model.add(Embedding(max_words, 100, input_length=max_len))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.summary()
model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])

#df=pd.read_csv('./testing.csv').astype("str")
df=pd.read_csv('./testing.csv')
x=df['tweet'].astype("str")
y=df['label']
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)

#print(x_test)

# tokenizer = Tokenizer(num_words=max_words)
# tokenizer.fit_on_texts(x_train,lower=False)

from keras.callbacks import EarlyStopping,ModelCheckpoint

stop = EarlyStopping(
    monitor='val_accuracy', 
    mode='max',
    patience=5
)

checkpoint= ModelCheckpoint(
    filepath='./',
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

# Define Flower client
class CifarClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        train_sequences = tokenizer.texts_to_sequences(x_train)
        train_sequences_matrix = tf.keras.utils.pad_sequences(train_sequences,maxlen=max_len)
        #model.fit(x_train, y_train, epochs=1, batch_size=32)
        model.fit(train_sequences_matrix,y_train,batch_size=1024,epochs=1,validation_split=0.2,callbacks=[stop,checkpoint])
        return model.get_weights(), len(train_sequences_matrix), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        test_sequences = tokenizer.texts_to_sequences(x_test)
        test_sequences_matrix = tf.keras.utils.pad_sequences(test_sequences,maxlen=max_len)
        loss, accuracy = model.evaluate(test_sequences_matrix, y_test)
        return loss, len(test_sequences_matrix), {"accuracy": accuracy}


# Start Flower client
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=CifarClient())
