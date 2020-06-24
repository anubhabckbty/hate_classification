# importing the necessary libraries 

import sys 
import pandas as pd
import numpy as np
import tensorflow as tf 
import tensorflow_hub as hub
import preprocessor as p 
from keras import backend as k

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate, Lambda
from tensorflow.keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from tensorflow.keras.layers import LSTM, Dropout
from tensorflow.keras.preprocessing import  text, sequence
from tensorflow.keras.callbacks import Callback, ModelCheckpoint

# creating the preprocessing function 

# here text is the complete text corpus, assuming it is line by line, it can be a dataframe or a list
def preprocess_text(text):

	text = [p.clean(i) for i in text]
	text = np.array(text, dtype=object)[:, np.newaxis]

	return text 

# defining the elmo layer 

def ELMoEmbedding(input_text):
	elmo = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)

	return elmo(tf.reshape(tf.cast(input_text, tf.string), [-1]), signature="default", as_dict=True)["elmo"]


# defining the architecture 
def get_model():
	inp = Input(shape=(1,), dtype="string", name="Input_layer")
	embedding_layer = Lambda(ELMoEmbedding, output_shape=(300,), name="Elmo_embedding")(inp)
	x = SpatialDropout1D(0.2)(embedding_layer)
	x = Bidirectional(GRU(80, return_sequences=True))(x)

	# output1: classification
	y = Bidirectional(LSTM(80, return_sequences=False, recurrent_dropout=0.2, dropout=0.2))(x)
	y = Dense(720, activation='relu')(y)
	y = Dropout(0.3)(y)
	y = Dense(360, activation='relu')(y)
	y = Dropout(0.3)(y)
	outp1 = Dense(2, activation="softmax", name="classification")(y)

	#output2: regression 
	avg_pool = GlobalAveragePooling1D()(x)
	max_pool = GlobalMaxPooling1D()(x)
	conc = concatenate([avg_pool, max_pool])
	outp2 = Dense(1, activation='sigmoid', name='regression')(conc)

	model = Model(inputs=inp, outputs=[outp1, outp2])
	
	model.compile(loss=['sparse_categorical_crossentropy', 'mse'], optimizer='adam', metrics={'classification':'accuracy', 'regression':'mse'})

	return model


# get the predictions, pass the preprocessed text, model and filepath of the weights file 
def get_predictions(text, model, filepath):
	with tf.Session() as session:
		k.set_session(session)
		session.run(tf.global_variables_initializer())
		session.run(tf.tables_initializer())
		model.load_weights(filepath)
		predict_labels, predict_retweets = model.predict(text, verbose=1)

	predictions = []	
	for i in predict_labels:
		if i[0]> i[1]:
			predictions.append(0)
		else:
			predictions.append(1)
	return predictions
