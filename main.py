import ast
import io
from random import random

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas
import math

import pandas as pd
import tensorflow as tf
from keras.applications.densenet import layers
from keras.models import Sequential
from keras.layers import Dense, Embedding, Bidirectional, Dropout
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from aws.aws_db import AWSSearchDB
from trining_set_provider.corpus_utils import write_corpus_to_file, get_vocabulary, get_vocabulary_length, \
    training_set_one_hot_encodings, get_context_words, context_words_df_berteilung, get_context_words_berteilung


def one_hot_samples(input_, target):
    # onehot encode the inputs and the targets
    # Example:
    # if character 'd' is encoded as 3 and n_unique_chars = 5
    # result should be the vector: [0, 0, 0, 1, 0], since 'd' is the 4th character
    return tf.one_hot(input_, 74), tf.keras.layers.concatenate(tf.one_hot(target, 74), axis=2)


if __name__ == '__main__':
    # trainX = None
    # trainY = None
    #
    # f = open('/Users/danilamarius-cristian/PycharmProjects/pythonProject3/raw_index.json.bkp')
    # text = ''.join(x for x in f.read() if x.isalpha() or x == ' ')
    # f.close()
    #
    # BATCH_SIZE = 128
    # EPOCHS = 30
    #
    # n_chars = len(text)
    # vocab = ''.join(sorted(set(text)))
    # print("unique_chars:", vocab)
    # n_unique_chars = len(vocab)
    #
    # print("Number of characters:", n_chars)
    # print("Number of unique characters:", n_unique_chars)
    #
    # # dictionary that converts characters to integers
    # char2int = {c: i for i, c in enumerate(vocab)}
    # # dictionary that converts integers to characters
    # int2char = {i: c for i, c in enumerate(vocab)}
    #
    # f = open('/Users/danilamarius-cristian/PycharmProjects/pythonProject3/raw_index.json.bkp')
    #
    # # returns JSON object as
    # # a dictionary
    # data = ast.literal_eval(f.read())
    #
    # f.close()
    #
    # skip = True
    #
    # anamnesis1 = ''.join(x for x in data['hits']['hits'][0]['_source']['anamnesis'] if x.isalpha() or x == ' ')
    # ctx_wds_1 = ''.join(x for x in get_context_words(anamnesis1) if x.isalpha() or x == ' ')
    #
    # ctx_wds_1_int = [char2int[c] for c in ctx_wds_1]
    # anamnesis1_int = [char2int[c] for c in anamnesis1]
    #
    # ctx_wds_1_int += (256 - len(ctx_wds_1_int)) * [0]
    # anamnesis1_int += (256 - len(anamnesis1_int)) * [0]
    #
    # print(tf.one_hot(tf.constant(anamnesis1_int[:256]), 74))
    #
    # ds = tf.data.Dataset.from_tensors((tf.constant(ctx_wds_1_int), tf.constant(anamnesis1_int[:256])))
    #
    # count = 0
    #
    # for i in data['hits']['hits']:
    #
    #     if count == 2:
    #         break
    #
    #     if skip == True:
    #         skip = False
    #         continue
    #
    #     anamnesis = ''.join(x for x in i['_source']['anamnesis'] if x.isalpha() or x == ' ')
    #     if len(anamnesis) > 256:
    #         continue
    #
    #     try:
    #
    #         ctx_wds_1 = ''.join(x for x in get_context_words(anamnesis) if x.isalpha() or x == ' ')
    #
    #         ctx_wds_1_int = [char2int[c] for c in ctx_wds_1]
    #         anamnesis1_int = [char2int[c] for c in anamnesis]
    #
    #         ctx_wds_1_int += (256 - len(ctx_wds_1_int)) * [0]
    #         anamnesis1_int += (256 - len(anamnesis1_int)) * [0]
    #
    #         other_ds = tf.data.Dataset.from_tensors((tf.constant(ctx_wds_1_int), tf.constant(anamnesis1_int)))
    #         ds = ds.concatenate(other_ds)
    #
    #         count += 1
    #         print(20 * "===")
    #         print(count)
    #         print(20 * "===")
    #
    #         # print(50 * "=")
    #         # print(anamnesis)
    #         # print(ctx_wds)
    #         # print(50 * "=")
    #     except Exception as e:
    #         print(e)
    #
    # dataset = ds.map(one_hot_samples)
    # print(dataset)
    #
    # ds = dataset.repeat().shuffle(1024).batch(2)
    #
    # model = Sequential([
    #     LSTM(256, input_shape=(256, 74), return_sequences=True),
    #     Dropout(0.3),
    #     LSTM(256 * 74),
    #     Dense(256 * 74, activation="softmax"),
    # ])
    #
    # model_weights_path = f"results.h5"
    # model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    # model.summary()
    #
    # # model.fit(ds, steps_per_epoch=20) // BATCH_SIZE, epochs=EPOCHS)
    # model.fit(ds, epochs=1, steps_per_epoch=1)
    # model.save(model_weights_path)


    f = open('/Users/danilamarius-cristian/PycharmProjects/pythonProject3/raw_index.json.bkp')

    # returns JSON object as
    # a dictionary
    data = ast.literal_eval(f.read())

    f.close()

    raw_dct = {"input_text": [], "target_text": []}


    counter = 0
    for i in data['hits']['hits']:
        try:
            print("Proccessing: ", counter)
            anamnesis = ''.join(x for x in i['_source']['berteleug'] if x.isalpha() or x == ' ')
            ctx_wds_1 = '|'.join(x for x in get_context_words_berteilung(anamnesis) if x.isalpha() or x == ' ')
            raw_dct["input_text"].append(ctx_wds_1)
            raw_dct["target_text"].append(anamnesis)
            counter += 1
        except Exception as e:
            print(e)
            continue

        if counter == 500:
            break

    df = pd.DataFrame(raw_dct)
    df.to_csv('train_berteilung.csv')

