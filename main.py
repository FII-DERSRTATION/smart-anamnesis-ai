import keras
import matplotlib.pyplot as plt
import pandas
import math

import tf as tf
from keras.applications.densenet import layers
from keras.models import Sequential
from keras.layers import Dense, Embedding, Bidirectional
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from aws.aws_db import AWSSearchDB
from trining_set_provider.corpus_utils import write_corpus_to_file, get_vocabulary, get_vocabulary_length, \
    training_set_one_hot_encodings

if __name__ == '__main__':

    trainX = None
    trainY = None

    # initial network
    # model = Sequential()
    # model.add(LSTM(4, input_shape=(1, look_back)))
    # model.add(Dense(1))
    # model.compile(loss='mean_squared_error', optimizer='adam')
    # model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

    vocab_size = get_vocabulary_length()

    model = Sequential()

    model.add(layers.Embedding(input_dim=vocab_size, output_dim=64))

    model.add(Bidirectional(LSTM(256, activation='relu')))

    model.add(Dense(vocab_size, activation='softmax'))

    model.compile(loss=keras.losses.cosine_similarity)

    model.fit()


    # awsdb = AWSSearchDB()
    # awsdb.export_anamnesis_to_file("raw_index.json")

    # write_corpus_to_file()
    # get_vocabulary()
    # print(get_vocabulary_length())
    # define example
    training_set_one_hot_encodings()
