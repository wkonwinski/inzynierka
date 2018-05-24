import numpy as np
import tensorflow as tf
import pickle
from keras.layers.recurrent import LSTM
from keras.layers import Dense
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras import backend as kerback
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, History
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

fileName = 'news'
path = r'../SharedData/'

with open(path + fileName + '_unchanged.txt', encoding = 'utf-8') as f:
    sourceText = f.read().lower()

######################## TWORZENIE ZESTAWU TRENINGOWEGO  #################
tokenizer = Tokenizer(lower = True, char_level = True, filters='')
tokenizer.fit_on_texts([sourceText])
lettersNumber = len(tokenizer.word_index) + 1

seqLen = 15
trainingPairs = []
for i in range(seqLen, len(sourceText)):
    tmpSeq = ''.join(sourceText[i - seqLen:i + 1])
    encodedSeq = tokenizer.texts_to_sequences([tmpSeq])[0]
    trainingPairs.append(encodedSeq)  
    
trainingPairs = np.array(trainingPairs)
trainXLetters = trainingPairs[:, :-1]
trainYLetters = trainingPairs[:, -1]
trainYLetters = to_categorical(trainYLetters, num_classes = lettersNumber)

######################## USTAWIENIA GPU ##################
config = tf.ConfigProto() 
config.gpu_options.allow_growth = True
kerback.tensorflow_backend.set_session(tf.Session(config = config))

######################## MODEL  ##################
hiddenSize = 500
lstmDropout = 0.2
learningRate = 0.001
batchSize = 128
NNLettersModel = Sequential()
NNLettersModel.add(Embedding(input_dim = lettersNumber, output_dim = lettersNumber, input_length = seqLen, embeddings_initializer = 'identity', trainable = False))
NNLettersModel.add(LSTM(hiddenSize, dropout = lstmDropout, recurrent_dropout = lstmDropout, return_sequences = True))
NNLettersModel.add(LSTM(hiddenSize, dropout = lstmDropout, recurrent_dropout = lstmDropout, return_sequences = True))
NNLettersModel.add(LSTM(hiddenSize, dropout = lstmDropout, recurrent_dropout = lstmDropout))  
NNLettersModel.add(Dense(lettersNumber, activation = 'softmax'))
print(NNLettersModel.summary())
     
optimizer = Adam(lr = learningRate)
NNLettersModel.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
filePath = r'./' + fileName + '-letters.hdf5'
checkpoint = ModelCheckpoint(filePath, monitor = 'val_loss', verbose = 1, save_weights_only = True)
history = History()
callbacksList = [checkpoint, history]

NNLettersModel.fit(trainXLetters, trainYLetters, batchSize, epochs = 1000, validation_split = 0.1, verbose = 2, callbacks = callbacksList)

#for layer in NNLettersModel.layers:
#    weights = layer.get_weights()
#    break

def SaveHistory(history, fileName):
    with open(fileName + '_historyLetters.txt', 'wb') as f:
        pickle.dump(history.history, f)
saveHist = SaveHistory(history, fileName)

kerback.clear_session()
del(NNLettersModel)


    