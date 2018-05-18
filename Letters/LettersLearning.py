import numpy as np
import tensorflow as tf
import pickle
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers import Dense
from keras.models import Sequential
from keras import backend as kerback
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, History
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

fileName = 'Pan_Tadeusz'
path = r'../SharedData/'

with open(path + fileName + '_unchanged.txt', encoding = 'utf-8') as f:
    sourceText = f.read().lower()

######################## TWORZENIE ZESTAWU TRENINGOWEGO  #################
tokenizer = Tokenizer(lower = True, char_level = True, filters='')
tokenizer.fit_on_texts([sourceText])
lettersNumber = len(tokenizer.word_index) + 1

seqLen = 25
trainingPairs = []
#for i in range(100, 10 ** 4 + 100):
for i in range(seqLen, len(sourceText)):
    tmpSeq = ''.join(sourceText[i - seqLen:i + 1])
    encodedSeq = tokenizer.texts_to_sequences([tmpSeq])[0]
    trainingPairs.append(encodedSeq) 

if fileName == 'Pan_Tadeusz':
    trainingPairs = trainingPairs[:10**5 + 5 * 10 ** 4]
    
    
trainingPairs = np.array(trainingPairs)
trainXLetters = trainingPairs[:, :-1]
trainYLetters = trainingPairs[:, -1]
trainYLetters = to_categorical(trainYLetters, num_classes = lettersNumber)


######################## USTAWIENIA GPU ##################
config = tf.ConfigProto() 
config.gpu_options.allow_growth = True
kerback.tensorflow_backend.set_session(tf.Session(config = config))

######################## MODEL  ##################
hiddenSize = 800
lstmDropout = 0.5
learningRate = 0.001
batchSize = 256

NNLettersModel = Sequential()
NNLettersModel.add(Embedding(input_dim = lettersNumber, output_dim = 10, input_length = seqLen))
NNLettersModel.add(LSTM(hiddenSize, dropout = lstmDropout, recurrent_dropout = lstmDropout, return_sequences = True))
NNLettersModel.add(LSTM(hiddenSize, dropout = lstmDropout, recurrent_dropout = lstmDropout, return_sequences = True))
NNLettersModel.add(LSTM(hiddenSize, dropout = lstmDropout, recurrent_dropout = lstmDropout))  
NNLettersModel.add(Dense(lettersNumber, activation = 'softmax'))
print(NNLettersModel.summary())
     
optimizer = Adam(lr = learningRate)

NNLettersModel.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics = ['acc'])

filePath = r'./' + fileName + '-letters.hdf5'
checkpoint = ModelCheckpoint(filePath, monitor = 'acc', verbose = 1, save_weights_only = True)
history = History()
callbacksList = [checkpoint, history]

NNLettersModel.fit(trainXLetters, trainYLetters, batchSize, epochs = 100, validation_split = 0.1, verbose = 2, callbacks = callbacksList)


def SaveHistory(history, fileName):
    with open(fileName + '_historyLetters.txt', 'wb') as f:
        pickle.dump(history.history, f)
saveHist = SaveHistory(history, fileName)

kerback.clear_session()
del(NNLettersModel)

    