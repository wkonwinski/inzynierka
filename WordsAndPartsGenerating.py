import numpy as np
import re
import requests
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import backend as kerback
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

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

    parameters = np.loadtxt(pathResources + fileName + '_parameters.txt', dtype = int)
    partsOfSpeechNumber = parameters[0]
    vocabSize = parameters[1]
    embeddingDim = parameters[2]

    dataPartOfSpeechAsString = ' '.join(dataPartOfSpeech)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([dataPartOfSpeechAsString])
    
#    config = tf.ConfigProto()
#    config.gpu_options.allow_growth = True
#    kerback.tensorflow_backend.set_session(tf.Session(config = config))
    ######################## MODEL GENERUJACY SLOWA ##################
    hiddenSize = 500
    learningRate = 0.001
          
    NNWordModel = Sequential()
    NNWordModel.add(Embedding(input_dim = vocabSize, output_dim = embeddingDim, input_length = wSeqLen))
    NNWordModel.add(LSTM(hiddenSize, return_sequences = True))
    NNWordModel.add(LSTM(hiddenSize, return_sequences = True))
    NNWordModel.add(LSTM(hiddenSize))
    NNWordModel.add(Dense(vocabSize, activation = 'softmax'))
    
    optimizer = Adam(lr = learningRate)
    NNWordModel.compile(loss = 'sparse_categorical_crossentropy', optimizer = optimizer, metrics = ['acc'])
    NNWordModel.load_weights(pathResources + fileName + '-wap-words.hdf5')

    ######################## MODEL GENERUJACY CZESCI MOWY ##################
    NNPartOfSpeechModel = Sequential()
    NNPartOfSpeechModel.add(Embedding(input_dim = partsOfSpeechNumber, output_dim = partsOfSpeechNumber, input_length = pSeqLen, embeddings_initializer = 'identity', trainable = False))
    NNPartOfSpeechModel.add(LSTM(hiddenSize, return_sequences = True)) 
    NNPartOfSpeechModel.add(LSTM(hiddenSize, return_sequences = True)) 
    NNPartOfSpeechModel.add(LSTM(hiddenSize))
    NNPartOfSpeechModel.add(Dense(partsOfSpeechNumber, activation = 'softmax'))
    
    optimizer = Adam(lr = learningRate)
    NNPartOfSpeechModel.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics = ['acc'])
    NNPartOfSpeechModel.load_weights(pathResources + fileName + '-wap-parts.hdf5')

    output = Generate(inputText, sourceText, sentenceLength, vocabSize, wSeqLen, pSeqLen, newIndexToWordTable, tokenizer, NNPartOfSpeechModel, NNWordModel)
    return output


def Word2Index(word, vocabSize, newIndexToWordTable):
    for i in range(0, vocabSize):
        if newIndexToWordTable[i].lower() == word.lower():
            return i
    return '@err'

def Index2Word (index, vocabSize, newIndexToWordTable):
    if newIndexToWordTable[index] != None:
        return newIndexToWordTable[index]
    else:
        return '@err'

def Word2PartOfSpeech(inputWord):
    session = requests.Session()
    retry = Retry(connect = 3, backoff_factor = 0.5)
    adapter = HTTPAdapter(max_retries = retry)
    session.mount('http://', adapter)
    urlHead = r'http://clarin.pelcra.pl/apt_pl/?sentences=["'
    urlTail = r'"]'
    try:
        req = session.get(urlHead + inputWord + urlTail)
    except requests.exceptions.RequestException as e:
        print('Blad polaczenia')
    reqVal = req.json()
    reqVal = reqVal['sentences'][0]
    for k, v in reqVal[0].items():
        if k == 'udt':
            return v.lower()
    return '@err'
#    for i in range(0, len(sourceText)):
#        if sourceText[i].lower() == inputWord.lower() or sourceText[i].lower() == inputWord.lower() + '.':
#            return dataPartOfSpeech[i]
#    return '@err'


def Generate(text, sourceText, wordsNumber, vocabSize, wSeqLen, pSeqLen, newIndexToWordTable, tokenizer, NNPartOfSpeechModel, NNWordModel):
    output = text
    partsOfSpeech = ''
    wordsIndexes = ''
    for word in text.lower().split():
        tmpResult = Word2Index(word, vocabSize, newIndexToWordTable)
        if tmpResult == '@err':
            return 'Brak podanego słowa: ' + word
        wordsIndexes += ' ' + str(tmpResult)
        partsOfSpeech += ' ' + Word2PartOfSpeech(word)

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

        if partOfSpeech == 'eol':
            output += '\n'
        elif partOfSpeech == 'dot':
            output += '. '
        elif partOfSpeech == 'com':
            output += ', '
        elif partOfSpeech == 'exc':
            output += '! '
        elif partOfSpeech == 'que':
            output += '? '
        elif partOfSpeech == 'eoa':
            output += '<eoa>'
        elif partOfSpeech == 'col':
            output += ': '
        elif partOfSpeech == 'sem':
            output += '; '
        elif partOfSpeech == 'quo':
            output += r'"'
        elif partOfSpeech == 'das':
            output += ' - '
        elif partOfSpeech == 'obr':
            output += r' ('
        elif partOfSpeech == 'cbr':
            output += r') '
        elif partOfSpeech == 'num':
            output += ' <num> '
        elif partOfSpeech == 'ast':
            output += '*'
        elif partOfSpeech == 'lar':
            output += ' «'
        elif partOfSpeech == 'rar':
            output += '» '          
                                  
        else:
            paddedSeq = pad_sequences([wordsIndexes.split()], maxlen = wSeqLen, padding = 'pre')
            predictedProbability = NNWordModel.predict(paddedSeq).flatten()
            sortedIndexes = np.argsort(predictedProbability)
            predsAndIndexes = []
            for i in range(len(predictedProbability)):
                row = []
                row.append(predictedProbability[i])
                row.append(sortedIndexes[i])
                predsAndIndexes.append(row)                            
            predsAndIndexes = sorted(predsAndIndexes, key = lambda x: x[0])[::-1]                    

            for i in range(len(predsAndIndexes)):
                word = Index2Word(predsAndIndexes[i][1], vocabSize, newIndexToWordTable)
                if word == '@err':
                    return 'err - w'
                part = Word2PartOfSpeech(word)
                if part == '@err':
                    return 'err - p'
                if part  == partOfSpeech:
                    output += ' ' + word
                    wordsIndexes += ' ' + str(predsAndIndexes[i][1])
                    break
    kerback.clear_session()
    del(NNPartOfSpeechModel)
    del(NNWordModel)
    return output 
                       

#text = ''
#sentenceLength = 200
#fileName = 'Pan_Tadeusz'
#pSeqLen = 10
#wSeqLen = 10
#test = GenerateSentenceWaP(text, sentenceLength, wSeqLen, pSeqLen, fileName)


# test2 = Word2Index('pijan', vocabSize, newIndexToWordTable)
#
#
#
# test3=Index2Word (15632, vocabSize, newIndexToWordTable)
#
# print(dataPartOfSpeech[4539])
# print(newIndexToWordTable[4539])

        



