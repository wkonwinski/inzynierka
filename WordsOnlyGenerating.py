import numpy as np
import re
import tensorflow as tf
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras import backend as kerback

def GenerateSentenceWordOnly(inputText, sentenceLength, temperature, seqLength, fileName):
    pathResources = r'./OnlyWords/'
    pattern = re.compile("([a-zA-ZęóąśłńćźżĘÓĄŚŁŃĆŻŹ&0.,!?;:-]+|\n)")
    with open(pathResources + fileName + '_newIndexToWordTable.txt') as f:
        newIndexToWordTable = re.findall(pattern, f.read().lower())

#    weightsForWordsInInputText = np.load(pathResources + fileName + '_weightsForWordsInInputText.npy')

    parameters = np.loadtxt(pathResources + fileName + '_parameters.txt', dtype = int)
    vocabSize = parameters[0]
    embeddingDim = parameters[1]


    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    kerback.tensorflow_backend.set_session(tf.Session(config = config))
    ######################## MODEL GENERUJACY SLOWA ##################
    hiddenSize = 600
    learningRate = 0.001
          
    NNWordOnlyModel = Sequential()
    NNWordOnlyModel.add(Embedding(input_dim = vocabSize, output_dim = embeddingDim, input_length = seqLength))
    NNWordOnlyModel.add(LSTM(hiddenSize, return_sequences = True))
    NNWordOnlyModel.add(LSTM(hiddenSize, return_sequences = True))
    NNWordOnlyModel.add(LSTM(hiddenSize))  
    NNWordOnlyModel.add(Dense(vocabSize, activation = 'softmax'))
    
    optimizer = Adam(lr = learningRate)
    NNWordOnlyModel.compile(loss = 'sparse_categorical_crossentropy', optimizer = optimizer, metrics = ['acc'])
    NNWordOnlyModel.load_weights(pathResources + fileName + '-words.hdf5')

    output = Generate(inputText, sentenceLength, temperature, seqLength, vocabSize, newIndexToWordTable, NNWordOnlyModel)
    return output


def Word2Index(word, vocabSize, newIndexToWordTable):
    for i in range(0, vocabSize):
        if newIndexToWordTable[i].lower() == word.lower():
            return i
    return -1


def Index2Word (index, vocabSize, newIndexToWordTable):
    for i in range (0, vocabSize):
        if i == index:
            return newIndexToWordTable[i]
    return -1


def GetPredictions(predictions, temperature):
    if temperature <= 0:
        return np.argmax(predictions)
    predictions = np.asarray(predictions).astype('float64')
    predictions = np.log(predictions) / temperature
    exponentialPredictions = np.exp(predictions)
    predictions = exponentialPredictions / np.sum(exponentialPredictions)
    probabilities = np.random.multinomial(1, predictions, 1)
    return np.argmax(probabilities)


def Generate(text, wordsNumber, temperature, seqLength, vocabSize, newIndexToWordTable, NNWordOnlyModel):
    wordsIndexes = [Word2Index(word, vocabSize, newIndexToWordTable) for word in text.lower().split()]
    for i in range(0, wordsNumber):
        reshapedArray = pad_sequences([wordsIndexes], maxlen = seqLength, padding = 'pre')
        prediction = NNWordOnlyModel.predict(reshapedArray)
        index = GetPredictions(prediction[-1], temperature)
        wordsIndexes.append(index)
    return ' '.join(Index2Word(idx, vocabSize, newIndexToWordTable) for idx in wordsIndexes)


inputText = 'Chybaby nie wiedziała , co znaczy twarz blada'
sentenceLength = 300
temperature = 0
fileName = 'Fraszki'
seqLen = 30
test = GenerateSentenceWordOnly(inputText, sentenceLength, temperature, seqLen, fileName)




