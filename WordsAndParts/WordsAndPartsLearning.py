import numpy as np
import gensim
import re
import tensorflow as tf
import pickle
import requests
import time
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers import Dense
from keras.models import Sequential
from keras import backend as kerback
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, History
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

session = requests.Session()
retry = Retry(connect = 3, backoff_factor = 0.5)
adapter = HTTPAdapter(max_retries = retry)
session.mount('http://', adapter)
urlHead = r'http://clarin.pelcra.pl/apt_pl/?sentences=["'
urlTail = r'"]'

fileName = 'Wesele'
path = r'../SharedData/'
modelWord2Vec = gensim.models.KeyedVectors.load_word2vec_format(path + 'nkjp+wiki-forms-all-100-skipg-hs.txt.gz') 
embeddingDim = 100

pattern = re.compile("([a-zA-ZęóąśłńćźżĘÓĄŚŁŃĆŻŹ]+)")
with open(path + fileName + '.txt', encoding = 'utf-8') as f:
    sourceTextTmp = re.findall(pattern, f.read().lower())

sourceText = []   
for word in sourceTextTmp:
    if word != 'num' and word != 'eoa':
        sourceText.append(word)
    
pattern = re.compile("([a-zA-Z0-9ęóąśłńćźżĘÓĄŚŁŃĆŻŹ.,*«»:;!?\<\>\"\(\)\-]+|\n)")
with open(fileName + '_PoS.txt') as f:
    dataPartOfSpeech = re.findall(pattern, f.read().lower())

###########################################
unkVector = (np.random.random_sample((1, 100)) * 2) - 1

seqenceOfWordsConvertedToModelIndexes = []
i = 0
while i < len(sourceText):    
    if sourceText[i] in modelWord2Vec.vocab:
        seqenceOfWordsConvertedToModelIndexes.append(modelWord2Vec.vocab[sourceText[i]].index)
    elif sourceText[i].title() in modelWord2Vec.vocab:
        seqenceOfWordsConvertedToModelIndexes.append(modelWord2Vec.vocab[sourceText[i].title()].index)
    else:
        basicForm = ''
        try:
            req = session.get(urlHead + sourceText[i] + urlTail)
        except requests.exceptions.RequestException as e:
            print(i)
            print('Ide spac')
            time.sleep(5)            
            continue
        reqVal = req.json()
        reqVal = reqVal['sentences'][0]
        for k, v in reqVal[0].items():
            if k == 'l':
                basicForm = basicForm.split('_')
                break
        for word in basicForm:
            if len(word) > 0:
                if word in modelWord2Vec.vocab:
                    seqenceOfWordsConvertedToModelIndexes.append(modelWord2Vec.vocab[word].index)
                elif word.title() in modelWord2Vec.vocab:
                    seqenceOfWordsConvertedToModelIndexes.append(modelWord2Vec.vocab[word.title()].index)
                else:
                    print('Nie ma:')
                    print(word)
                    seqenceOfWordsConvertedToModelIndexes.append(-1)                          
    i += 1          

##############################################

#bierzemy tylko te słowa i ich wektory, które znajdują się w tekcie, dlatego że tylko to potrzebne jest do uczenia i generowania
# index tablicy -> [index słowa w modelu word2vec]
oldToNewIndexTable = sorted(set(seqenceOfWordsConvertedToModelIndexes)) 
vocabSize = len(oldToNewIndexTable)
# wagi dla wszystkich słów z modelu
allPretrainedWeights = modelWord2Vec.vectors
# tablica na wagi słów, które znajdują się w tekscie 
weightsForWordsInInputText = np.zeros(shape = (vocabSize, embeddingDim))
newIndexToWordTable = []
for i in range(0, vocabSize):
    # tablica słów odpowiadających nowym indexom 
    #  model w2v    |   Nowy model -> Model w2v |    Nowy model  
    # idx   slowo   |    idx           idx      |  idx     slowo
    #  5      a     |     1             5       |   1        a
    #  17     b     |     2             17      |   2        b  
    #  45     c     |     3             45      |   3        c 
    if oldToNewIndexTable[i] == -1:
         newIndexToWordTable.append('<UNK>')
         weightsForWordsInInputText[i] = unkVector
    else:
        newIndexToWordTable.append(modelWord2Vec.index2word[oldToNewIndexTable[i]])   
        #kopiowanie wektora danego słowa do nowej tablicy pod odpowiedni index
        for j in range(0, embeddingDim):
            weightsForWordsInInputText[i, j] = allPretrainedWeights[oldToNewIndexTable[i], j]
 
       
seqenceOfWordsConvertedToNewIndexes = []
# konwesja ciągu starych indexów na nowe indexy: 5 17 45. ==> 1 2 3.
for wordIndex in seqenceOfWordsConvertedToModelIndexes:
    for i in range(0, vocabSize):
        if wordIndex == oldToNewIndexTable[i]:
                seqenceOfWordsConvertedToNewIndexes.append(i)
                break
              
######################## TWORZENIE ZESTAWU TRENINGOWEGO DLA MODELU UCZACEGO SIE SLOW #################
wSeqLen = 10
trainingPairs = []
for i in range(wSeqLen, len(seqenceOfWordsConvertedToNewIndexes)):
    trainingPairs.append(seqenceOfWordsConvertedToNewIndexes[i - wSeqLen:i + 1])
       
trainingPairs = np.array(trainingPairs) 
trainXWord = trainingPairs[:, :-1]
trainYWord = trainingPairs[:, -1]

######################## TWORZENIE ZESTAWU TRENINGOWEGO DLA MODELU UCZACEGO SIE CZESCI MOWY #################
dataPartOfSpeechAsString = ' '.join(dataPartOfSpeech)
tokenizer = Tokenizer(filters='')
tokenizer.fit_on_texts([dataPartOfSpeechAsString])
partsOfSpeechNumber = len(tokenizer.word_index) + 1         

pSeqLen = 10
trainingPairs = []
for i in range(pSeqLen, len(dataPartOfSpeech)):
    tmpSeq = ' '.join(dataPartOfSpeech[i - pSeqLen:i + 1])
    encodedSeq = tokenizer.texts_to_sequences([tmpSeq])[0]
    trainingPairs.append(encodedSeq) 
    
trainingPairs = np.array(trainingPairs)
trainXPartOfSpeech = trainingPairs[:, :-1]
trainYPartOfSpeech = trainingPairs[:, -1]
trainYPartOfSpeech = to_categorical(trainYPartOfSpeech, num_classes = partsOfSpeechNumber)
########################ZAPISYWANIE DANYCH#########################
parametersToSave = []
parametersToSave.append(partsOfSpeechNumber)
parametersToSave.append(vocabSize)
parametersToSave.append(embeddingDim)

def saveData(data, name):
    tmpData = '' 
    for word in data:
        tmpData += ' ' + word 
    with open(fileName + name + '.txt', "w") as file:        
        file.write(tmpData)
    
saveData(newIndexToWordTable, '_newIndexToWordTable')
np.savetxt(fileName + '_parameters.txt', parametersToSave, fmt = '%d')

######################## USTAWIENIA GPU ##################
config = tf.ConfigProto() 
config.gpu_options.allow_growth = True
kerback.tensorflow_backend.set_session(tf.Session(config = config))

######################## PARAMETRY SIECI ###########################
hiddenSize = 500
lstmDropout = 0.2
learningRate = 0.001
batchSize = 128 
######################## MODEL UCZACY SIĘ SŁÓW ##################      
NNWordModel = Sequential()
NNWordModel.add(Embedding(input_dim = vocabSize, output_dim = embeddingDim, weights = [weightsForWordsInInputText], trainable = False, input_length = wSeqLen))
NNWordModel.add(LSTM(hiddenSize, dropout = lstmDropout, recurrent_dropout = lstmDropout, return_sequences = True))
NNWordModel.add(LSTM(hiddenSize, dropout = lstmDropout, recurrent_dropout = lstmDropout, return_sequences = True))
NNWordModel.add(LSTM(hiddenSize, dropout = lstmDropout, recurrent_dropout = lstmDropout))
NNWordModel.add(Dense(vocabSize, activation = 'softmax'))
print(NNWordModel.summary())

optimizer = Adam(lr = learningRate)
NNWordModel.compile(loss = 'sparse_categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])

filePath = r'./' + fileName + '-wap-words.hdf5'
checkpoint = ModelCheckpoint(filePath, monitor = 'val_loss', verbose = 1, save_weights_only = True)
history = History()
callbacksList = [checkpoint, history]

NNWordModel.fit(trainXWord, trainYWord, batchSize, epochs = 1000, verbose = 2, validation_split = 0.1, callbacks = callbacksList, shuffle = True)

def SaveHistory(history, fileName):
    with open(fileName + '_historyWaPWords.txt', 'wb') as f:
        pickle.dump(history.history, f)
saveHist = SaveHistory(history, fileName)

kerback.clear_session()
del(NNWordModel)

######################## MODEL UCZACY SIĘ STRUKTURY ZDANIA ##################
     
NNPartOfSpeechModel = Sequential()
NNPartOfSpeechModel.add(Embedding(input_dim = partsOfSpeechNumber, output_dim = partsOfSpeechNumber, input_length = pSeqLen, embeddings_initializer = 'identity', trainable = False))
NNPartOfSpeechModel.add(LSTM(hiddenSize, dropout = lstmDropout, recurrent_dropout = lstmDropout, return_sequences = True))
NNPartOfSpeechModel.add(LSTM(hiddenSize, dropout = lstmDropout, recurrent_dropout = lstmDropout, return_sequences = True)) 
NNPartOfSpeechModel.add(LSTM(hiddenSize, dropout = lstmDropout, recurrent_dropout = lstmDropout))
NNPartOfSpeechModel.add(Dense(partsOfSpeechNumber, activation = 'softmax'))
print(NNPartOfSpeechModel.summary())

optimizer = Adam(lr = learningRate)
NNPartOfSpeechModel.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])

filePath = r'./' + fileName + '-wap-parts.hdf5'
checkpoint = ModelCheckpoint(filePath, monitor = 'val_loss', verbose = 1, save_weights_only = True)
history = History()
callbacksList = [checkpoint, history]

NNPartOfSpeechModel.fit(trainXPartOfSpeech, trainYPartOfSpeech, batchSize, epochs = 1000, verbose = 2, validation_split = 0.1, callbacks = callbacksList)

def SaveHistory(history, fileName):
    with open(fileName + '_historyWaPParts.txt', 'wb') as f:
        pickle.dump(history.history, f)
saveHist = SaveHistory(history, fileName)

kerback.clear_session()
del(NNPartOfSpeechModel)  
    
    