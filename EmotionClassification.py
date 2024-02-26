#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 12:54:54 2024

@author: machine
"""

import os 
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Dropout, Bidirectional,LSTM,Embedding
from tensorflow.keras.optimizers import RMSprop , Adam
from tensorflow.keras.metrics import Accuracy , F1Score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
import tensorflow as tf
import json

def remove_stopwords(text):
    text_split = text.split()
    words_nostop = [w for w in text_split if w not in stop_words]
    return " ".join(words_nostop)


#Directory for data and pre-trained GloVe embedding 
data_path = '/home/machine/Documents/project_files/Datasets/Emotion_data/data.csv'
work_dir = '/home/machine/Documents/project_files/NLP/EmotionClassification'
embedding_path = '/home/machine/Documents/project_files/NLP/GloVe/glove.6B.100d.txt'
train_split = 0.8
rand_state = 55

df = pd.read_csv(data_path, sep=',')
df = df[["text", "label"]]
df.head(10)
df.info()


class_dict = {0:'sadness', 1:'joy', 2:'love', 3:'anger', 4:'fear', 5:'surprise' }
classes = df["label"].unique()
classes = sorted(classes)
print(f"Unique labels: {classes}")
print("\nLabels' distribution")
df.hist("label")
stop_words = set(stopwords.words('english'))
df["text"]=df["text"].apply(remove_stopwords)



X_train, X_val, y_train, y_val = train_test_split(df["text"], df["label"],
                                                  train_size=train_split,
                                                  random_state=rand_state,
                                                  stratify=df["label"])

X_train = X_train.values.tolist()  # convert to list for later tokenization
X_val = X_val.values.tolist()  # convert to list for later tokenization
y_train = y_train.values
y_val = y_val.values



tokenizer = Tokenizer(oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)
word_ids = tokenizer.word_index 
print(f"Number of indexed words = {len(word_ids)}")
print ("Examples of Indexed words :/n", {k: word_ids[k] for k in list(word_ids)[:8]})



train_sequences = tokenizer.texts_to_sequences(X_train)
val_sequences = tokenizer.texts_to_sequences(X_val)
max_len = max([max([len(seq) for seq in train_sequences]),max([len(seq) for seq in val_sequences])])
X_train = pad_sequences(train_sequences, maxlen=max_len, padding='pre')
X_val = pad_sequences(val_sequences, maxlen=max_len, padding='pre')



all_embeddings = {}
with open(embedding_path) as file:
    for line in file:
        values = line.split()
        all_embeddings[values[0]] = np.asarray(values[1: ], dtype='float32')
print(f"There are {len(all_embeddings)} embedding words")
first_word = next(iter(all_embeddings))
print(f"Each word is an array with shape = {all_embeddings[first_word].shape}")



# Get the number of dimensions of each word
embedding_dim = all_embeddings[first_word].shape[0]
embedding_matrix = np.zeros((len(word_ids) + 1, embedding_dim))
for word, idx in word_ids.items():
    if all_embeddings.get(word) is not None:
        embedding_matrix[idx] = all_embeddings[word]



model1 = Sequential([
    Embedding(len(word_ids) + 1, embedding_dim, input_length=max_len,
              weights=[embedding_matrix], trainable=False),
    Dropout(0.1),
    Conv1D(128, 5, activation='relu'),
    MaxPooling1D(pool_size=5),
    Bidirectional(LSTM(64)),
    Dropout(0.1),
    Dense(512, activation='relu'),
    Dropout(0.1),
    Dense(len(classes), activation='softmax')
])



model1.compile(loss='sparse_categorical_crossentropy',
              optimizer=RMSprop(0.002),
              metrics=['accuracy'])
my_callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath='callback/ModelCheckpoint',
                                                            save_weights_only=False,verbose=1),
                tf.keras.callbacks.BackupAndRestore(backup_dir='callback/BackupandRestore',save_freq="epoch"),
                tf.keras.callbacks.TensorBoard(log_dir='callback/logs')]






saved_model_file = Path("weights/checkpoint")
if saved_model_file.exists():
    model1.load_weights('weights/EmC_weights')
    history_dict = json.load(open('history_model1', 'r'))
    print("Model Loaded")
else:
    history_model1 = model1.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val),callbacks=my_callbacks)
    model1.save_weights('weights/EmC_weights')
    history_dict = history_model1.history
    json.dump(history_dict,open('history_model1','w'))
    print("Model Saved")




epochs = [*range(20)]
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
plt.plot(epochs, acc, 'r')
plt.plot(epochs, val_acc, 'b')
plt.title('Training and validation accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["Accuracy", "Validation Accuracy"])
plt.show()




y_pred = model1.predict(X_val)
y_pred = np.argmax(y_pred, axis=1)
print("Classification report:\n", classification_report(y_val, y_pred))

