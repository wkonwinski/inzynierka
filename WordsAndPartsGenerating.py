import numpy as np
import re
import tensorflow as tf
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import backend as kerback

def GenerateSentenceWaP(inputText, sentenceLength, wSeqLen, pSeqLen, fileName):
    pathShared = r'./SharedData/'
    pathResources = r'./WordsAndParts/'
    pattern = re.compile("([a-zA-Z0-9ęóąśłńćźżĘÓĄŚŁŃĆŻŹ.,*«»:;!?\<\>\"\(\)\-]+|\n)")
    with open(pathShared + fileName + '.txt', encoding = 'utf-8') as f:
        sourceText = re.findall(pattern, f.read().lower())

    with open(pathResources + fileName + '_newIndexToWordTable.txt') as f:
        newIndexToWordTable = re.findall(pattern, f.read().lower())

    with open(pathResources + fileName + '_PoS.txt', encoding = 'utf-8') as f:
        dataPartOfSpeech = re.findall(pattern, f.read().lower())

#    weightsForWordsInInputText = np.load(pathResources + fileName + '_weightsForWordsInInputText.npy')

    parameters = np.loadtxt(pathResources + fileName + '_parameters.txt', dtype = int)
    partsOfSpeechNumber = parameters[0]
    vocabSize = parameters[1]
    embeddingDim = parameters[2]

    dataPartOfSpeechAsString = ' '.join(dataPartOfSpeech)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([dataPartOfSpeechAsString])
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    kerback.tensorflow_backend.set_session(tf.Session(config = config))
    # #
    ######################## MODEL GENERUJACY SLOWA ##################
    hiddenSize = 600
    lstmDropout = 0.3
    learningRate = 0.001
          
    NNWordModel = Sequential()
    NNWordModel.add(Embedding(input_dim = vocabSize, output_dim = embeddingDim, input_length = wSeqLen))
    NNWordModel.add(LSTM(hiddenSize, dropout = lstmDropout, recurrent_dropout = lstmDropout, return_sequences = True))
    NNWordModel.add(LSTM(hiddenSize, dropout = lstmDropout, recurrent_dropout = lstmDropout))
    NNWordModel.add(Dense(vocabSize, activation = 'softmax'))
    
    optimizer = Adam(lr = learningRate)
    NNWordModel.compile(loss = 'sparse_categorical_crossentropy', optimizer = optimizer, metrics = ['acc'])
    NNWordModel.load_weights(pathResources + fileName + '-wap-words.hdf5')

    ######################## MODEL GENERUJACY CZESCI MOWY ##################
    NNPartOfSpeechModel = Sequential()
    NNPartOfSpeechModel.add(Embedding(input_dim = partsOfSpeechNumber, output_dim = 10, input_length = pSeqLen))
    NNPartOfSpeechModel.add(LSTM(hiddenSize, dropout = lstmDropout, recurrent_dropout = lstmDropout, return_sequences = True)) 
    NNPartOfSpeechModel.add(LSTM(hiddenSize, dropout = lstmDropout, recurrent_dropout = lstmDropout))
    NNPartOfSpeechModel.add(Dense(partsOfSpeechNumber, activation = 'softmax'))
    
    optimizer = Adam(lr = learningRate)
    NNPartOfSpeechModel.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics = ['acc'])
    NNPartOfSpeechModel.load_weights(pathResources + fileName + '-wap-parts.hdf5')

    output = Generate(inputText, sourceText, dataPartOfSpeech, sentenceLength, vocabSize, wSeqLen, pSeqLen, newIndexToWordTable, tokenizer, NNPartOfSpeechModel, NNWordModel)
    return output



def Word2Index(word, vocabSize, newIndexToWordTable):
    for i in range(0, vocabSize):
        if newIndexToWordTable[i].lower() == word.lower():
            return i
    return '@err'


def Index2Word (index, vocabSize, newIndexToWordTable):
    for i in range (0, vocabSize):
        if i == index:
            return newIndexToWordTable[i]
    return '@err'


def Word2PartOfSpeech(inputWord, sourceText, dataPartOfSpeech):
    for i in range(0, len(sourceText)):
        if sourceText[i].lower() == inputWord.lower() or sourceText[i].lower() == inputWord.lower() + '.':
            return dataPartOfSpeech[i]
    return '@err'


def Generate(text, sourceText, dataPartOfSpeech, wordsNumber, vocabSize, wSeqLen, pSeqLen, newIndexToWordTable, tokenizer, NNPartOfSpeechModel, NNWordModel):
    output = text
    partsOfSpeech = ''
    wordsIndexes = ''
#    if len(text.lower().split()) == 0:
#        return 'Podaj slowo'

    for word in text.lower().split():
        tmpResult = Word2Index(word, vocabSize, newIndexToWordTable)
        if tmpResult == '@err':
            return 'Brak podanego słowa: ' + word
        wordsIndexes += ' ' + str(tmpResult)
        partsOfSpeech += ' ' + Word2PartOfSpeech(word, sourceText, dataPartOfSpeech)

    for i in range(0, wordsNumber):
        encoded = tokenizer.texts_to_sequences([partsOfSpeech])[0]
        encoded = pad_sequences([encoded], maxlen = pSeqLen, padding = 'pre')
        partOfSpeechPrediction = NNPartOfSpeechModel.predict_classes(encoded, verbose = 0)

        partOfSpeech = ''
        for part, index in tokenizer.word_index.items():
            if index == partOfSpeechPrediction:
                partOfSpeech = part
                break

        partsOfSpeech += ' ' + partOfSpeech

        if partOfSpeech == '<eol>':
            output += '\n'
        elif partOfSpeech == '<dot>':
            output += '. '
        elif partOfSpeech == '<com>':
            output += ', '
        elif partOfSpeech == '<exc>':
            output += '! '
        elif partOfSpeech == '<que>':
            output += '? '
        elif partOfSpeech == '<eoa>':
            output += '<eoa>'
        elif partOfSpeech == '<col>':
            output += ': '
        elif partOfSpeech == '<sem>':
            output += '; '
        elif partOfSpeech == '<quo>':
            output += r'"'
        elif partOfSpeech == '<das>':
            output += ' - '
        elif partOfSpeech == '<obr>':
            output += r' ('
        elif partOfSpeech == '<cbr>':
            output += r') '
        elif partOfSpeech == '<num>':
            output += ' <num> '
        elif partOfSpeech == '<ast>':
            output += '*'
        elif partOfSpeech == '<lar>':
            output += ' «'
        elif partOfSpeech == '<rar>':
            output += '» '          
                                  
        else:
            reshapedArray = pad_sequences([wordsIndexes.split()], maxlen = wSeqLen, padding = 'pre')
            predictedProbability = NNWordModel.predict(reshapedArray).flatten()
            sortedProbabilityValues = NNWordModel.predict(reshapedArray).flatten()
            sortedProbabilityValues[::-1].sort()
            find = 0
            for value in sortedProbabilityValues:
                for i in range(0, len(predictedProbability)):
                    if value == predictedProbability[i]:
                        word = Index2Word(i, vocabSize, newIndexToWordTable)
                        part = Word2PartOfSpeech(word, sourceText, dataPartOfSpeech)
                        if word == '@err':
                            print('word')
                            print(word)
                            print(i)
                            return 'Blad'
                        elif part == '@err':
                            print('part')
                            print(word)
                            print(i)
                        if part == partOfSpeech:
#                            if output.rstrip().split()[-1] == '.' or output.rstrip().split()[-1] == '!' or output.rstrip().split()[-1] == '?':
#                                    word = word.title()
                            output += ' ' + word
                            wordsIndexes += ' ' + str(i)
                            find = 1
                            break
                if find == 1:
                    break
    return output


# inputText = ''
# sentenceLength = 2
# fileName = 'Fraszki'
# pSeqLen = 25
# wSeqLen = 15
# test = GenerateSentenceWaP(inputText, sentenceLength, wSeqLen, pSeqLen, fileName)
#
#
# test2 = Word2Index('pijan', vocabSize, newIndexToWordTable)
#
#
# test3=Index2Word (test2, vocabSize, newIndexToWordTable)
#
# print(dataPartOfSpeech[4539])
# print(newIndexToWordTable[4539])





