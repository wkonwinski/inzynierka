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
from keras.optimizers import Adagrad
from keras.callbacks import ModelCheckpoint, History
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry


session = requests.Session()
retry = Retry(connect = 3, backoff_factor = 0.5)
adapter = HTTPAdapter(max_retries = retry)
session.mount('http://', adapter)
urlHead = r'http://clarin.pelcra.pl/apt_pl/?sentences=["'
urlTail = r'"]'

fileName = 'Pan_Tadeusz'
path = r'../SharedData/'
modelWord2Vec = gensim.models.KeyedVectors.load_word2vec_format(path + 'nkjp+wiki-forms-all-100-skipg-hs.txt.gz') 
embeddingDim = 100

pattern = re.compile("([a-zA-Z0-9ęóąśłńćźżĘÓĄŚŁŃĆŻŹ.,*«»:;!?\<\>\"\(\)\-]+|\n)")
with open(path + fileName + '.txt', encoding = 'utf-8') as f:
    sourceText = re.findall(pattern, f.read().lower())      
    
###########################################
dotVector = (np.random.random_sample((1, 100)) * 2) - 1
newLineVector = (np.random.random_sample((1, 100)) * 2) - 1
commaVector = (np.random.random_sample((1, 100)) * 2) - 1
exclVector = (np.random.random_sample((1, 100)) * 2) - 1
eoaVector = (np.random.random_sample((1, 100)) * 2) - 1
quesVector = (np.random.random_sample((1, 100)) * 2) - 1
colVector = (np.random.random_sample((1, 100)) * 2) - 1
quoVector = (np.random.random_sample((1, 100)) * 2) - 1
dasVector = (np.random.random_sample((1, 100)) * 2) - 1
obraVector = (np.random.random_sample((1, 100)) * 2) - 1
cbraVector = (np.random.random_sample((1, 100)) * 2) - 1
unkVector = (np.random.random_sample((1, 100)) * 2) - 1
semiVector = (np.random.random_sample((1, 100)) * 2) - 1
numVector = (np.random.random_sample((1, 100)) * 2) - 1
astVector = (np.random.random_sample((1, 100)) * 2) - 1
larVector = (np.random.random_sample((1, 100)) * 2) - 1
rarVector = (np.random.random_sample((1, 100)) * 2) - 1
#######################################################

seqenceOfWordsConvertedToModelIndexes = []
i = 0
pattern = re.compile("([a-zA-ZęóąśłńćźżĘÓĄŚŁŃĆŻŹ]+)")
while i < len(sourceText):
    if re.match(pattern, sourceText[i]):
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
                        seqenceOfWordsConvertedToModelIndexes.append(-12)            
    
    elif sourceText[i] == '.':
            seqenceOfWordsConvertedToModelIndexes.append(-1)
    elif sourceText[i] == ',':
            seqenceOfWordsConvertedToModelIndexes.append(-2)
    elif sourceText[i] == '!':
            seqenceOfWordsConvertedToModelIndexes.append(-3)
    elif sourceText[i] == '<eoa>':
            seqenceOfWordsConvertedToModelIndexes.append(-4)
    elif sourceText[i] == '?':
            seqenceOfWordsConvertedToModelIndexes.append(-5)
    elif sourceText[i] == ':':
            seqenceOfWordsConvertedToModelIndexes.append(-6)
    elif sourceText[i] == r'"':
            seqenceOfWordsConvertedToModelIndexes.append(-7)
    elif sourceText[i] == '-':
            seqenceOfWordsConvertedToModelIndexes.append(-8)
    elif sourceText[i] == r')':
            seqenceOfWordsConvertedToModelIndexes.append(-9)
    elif sourceText[i] == r'(':
            seqenceOfWordsConvertedToModelIndexes.append(-10)
    elif sourceText[i] == '\n':
            seqenceOfWordsConvertedToModelIndexes.append(-11)
    elif sourceText[i] == ';':
            seqenceOfWordsConvertedToModelIndexes.append(-13)
    elif sourceText[i] == '<num>':
            seqenceOfWordsConvertedToModelIndexes.append(-14)
    elif sourceText[i] == '*':
            seqenceOfWordsConvertedToModelIndexes.append(-15)
    elif sourceText[i] == '«':
            seqenceOfWordsConvertedToModelIndexes.append(-16)
    elif sourceText[i] == '»':
            seqenceOfWordsConvertedToModelIndexes.append(-17)
    else:
        print('Nie ma bardziej:')
        print(sourceText[i])
        seqenceOfWordsConvertedToModelIndexes.append(-12)               
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
         newIndexToWordTable.append('.')
         weightsForWordsInInputText[i] = dotVector
    elif oldToNewIndexTable[i] == -2:
        newIndexToWordTable.append(',')
        weightsForWordsInInputText[i] = commaVector
    elif oldToNewIndexTable[i] == -3:
        newIndexToWordTable.append('!')
        weightsForWordsInInputText[i] = exclVector
    elif oldToNewIndexTable[i] == -4:
        newIndexToWordTable.append('?')
        weightsForWordsInInputText[i] = quesVector  
    elif oldToNewIndexTable[i] == -5:
        newIndexToWordTable.append('<eoa>')
        weightsForWordsInInputText[i] = eoaVector
    elif oldToNewIndexTable[i] == -6:
        newIndexToWordTable.append(':')
        weightsForWordsInInputText[i] = colVector
    elif oldToNewIndexTable[i] == -7:
        newIndexToWordTable.append('\"')
        weightsForWordsInInputText[i] = quoVector
    elif oldToNewIndexTable[i] == -8:
        newIndexToWordTable.append('-')
        weightsForWordsInInputText[i] = dasVector
    elif oldToNewIndexTable[i] == -9:
        newIndexToWordTable.append('\)')
        weightsForWordsInInputText[i] = cbraVector
    elif oldToNewIndexTable[i] == -10:
        newIndexToWordTable.append('\(')
        weightsForWordsInInputText[i] = obraVector
    elif oldToNewIndexTable[i] == -11:
        newIndexToWordTable.append('\n')
        weightsForWordsInInputText[i] = newLineVector
    elif oldToNewIndexTable[i] == -12:
        newIndexToWordTable.append('<unk>')
        weightsForWordsInInputText[i] = unkVector
    elif oldToNewIndexTable[i] == -13:
        newIndexToWordTable.append(';')
        weightsForWordsInInputText[i] = semiVector
    elif oldToNewIndexTable[i] == -14:
        newIndexToWordTable.append('<num>')
        weightsForWordsInInputText[i] = numVector
    elif oldToNewIndexTable[i] == -15:
        newIndexToWordTable.append('*')
        weightsForWordsInInputText[i] = astVector
    elif oldToNewIndexTable[i] == -16:
        newIndexToWordTable.append('«')
        weightsForWordsInInputText[i] = larVector
    elif oldToNewIndexTable[i] == -17:
        newIndexToWordTable.append('»')
        weightsForWordsInInputText[i] = rarVector
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
              
#test = modelWord2Vec.index2word[309]
#test2 = newIndexToWordTable[237]


######################## TWORZENIE ZESTAWU TRENINGOWEGO DLA MODELU UCZACEGO SIE SLOW #################
trainingPairs = []
seqLen = 15
for i in range(seqLen, len(seqenceOfWordsConvertedToNewIndexes)):
    trainingPairs.append(seqenceOfWordsConvertedToNewIndexes[i - seqLen:i + 1])
       
trainingPairs = np.array(trainingPairs) 
trainXWord = trainingPairs[:, :-1]
trainYWord = trainingPairs[:, -1]

########################ZAPISYWANIE DANYCH#########################
parametersToSave = []
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

######################## MODEL UCZACY SIĘ SŁÓW ##################
hiddenSize = 800
lstmDropout = 0.5
learningRate = 0.01
batchSize = 256 
      
NNWordOnlyModel = Sequential()
NNWordOnlyModel.add(Embedding(input_dim = vocabSize, output_dim = embeddingDim, input_length = seqLen, weights = [weightsForWordsInInputText]))
NNWordOnlyModel.add(LSTM(hiddenSize, dropout = lstmDropout, recurrent_dropout = lstmDropout, return_sequences = True))
NNWordOnlyModel.add(LSTM(hiddenSize, dropout = lstmDropout, recurrent_dropout = lstmDropout, return_sequences = True))
NNWordOnlyModel.add(LSTM(hiddenSize, dropout = lstmDropout, recurrent_dropout = lstmDropout))  
NNWordOnlyModel.add(Dense(vocabSize, activation = 'softmax'))
print(NNWordOnlyModel.summary())

optimizer = Adagrad(lr = learningRate)
NNWordOnlyModel.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics = ['acc'])

filePath = r'./' + fileName + '-words.hdf5'
checkpoint = ModelCheckpoint(filePath, monitor = 'acc', verbose = 1, save_weights_only = True)
history = History()
callbacksList = [checkpoint, history]

NNWordOnlyModel.fit(trainXWord, trainYWord, batchSize, epochs = 100, verbose = 2, validation_split = 0.1, callbacks = callbacksList)

def SaveHistory(history, fileName):
    with open(fileName + '_historyWordsOnly.txt', 'wb') as f:
        pickle.dump(history.history, f)
saveHist = SaveHistory(history, fileName)

#kerback.clear_session()
#del(NNWordOnlyModel)
   
    