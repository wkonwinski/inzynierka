import numpy as np
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import backend as kerback

def GenerateLetters(inputText, sentenceLength, temperature, seqLength, fileName):
    fileName2 = fileName + '_unchanged'
    pathShared = r'./SharedData/'
    pathResources = r'./Letters/'
    
    with open(pathShared + fileName2 + '.txt', encoding = 'utf-8') as f:
        sourceText = f.read().lower()      

    tokenizer = Tokenizer(lower = False, char_level = True, filters='')
    tokenizer.fit_on_texts([sourceText])
    lettersNumber = len(tokenizer.word_index) + 1
#    ######################## USTAWIENIA GPU ##################
#    config = tf.ConfigProto()
#    config.gpu_options.allow_growth = True
#    kerback.tensorflow_backend.set_session(tf.Session(config = config))
    
    ######################## MODEL  ##################
    hiddenSize = 500
    learningRate = 0.001
    
    NNLettersModel = Sequential()
    NNLettersModel.add(Embedding(input_dim = lettersNumber, output_dim = lettersNumber, input_length = seqLength, embeddings_initializer = 'identity'))
    NNLettersModel.add(LSTM(hiddenSize, return_sequences = True, unroll = True))
    NNLettersModel.add(LSTM(hiddenSize, return_sequences = True, unroll = True))
    NNLettersModel.add(LSTM(hiddenSize, unroll = True))  
    NNLettersModel.add(Dense(lettersNumber, activation = 'softmax'))
         
    optimizer = Adam(lr = learningRate)
    
    NNLettersModel.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
    NNLettersModel.load_weights(pathResources + fileName + '-letters.hdf5')
    
    output = Generate(inputText, tokenizer, sentenceLength, NNLettersModel, temperature, seqLength, lettersNumber)    
    return output
    

def GetPredictions(predictions, temperature):
    if temperature <= 0:
        return np.argmax(predictions)
    predictions = np.asarray(predictions).astype('float64')
    predictions = np.log(predictions) / temperature
    exponentialPredictions = np.exp(predictions)
    predictions = exponentialPredictions / np.sum(exponentialPredictions)
    probabilities = np.random.multinomial(1, predictions, 1)
    return np.argmax(probabilities)

def Generate(inputText, tokenizer, sentenceLength, NNLettersModel, temp, seqLength, lettersNumber):
    i = 0
    while i < sentenceLength:
        cutText = inputText[-seqLength - 1:]
        encoded = tokenizer.texts_to_sequences([cutText])[0]
        encoded = pad_sequences([encoded], maxlen = seqLength, padding = 'pre')
        letterPrediction = NNLettersModel.predict(encoded)
        letterIndex = GetPredictions(letterPrediction[-1], temp)
        for char, index in tokenizer.word_index.items():
            if index == letterIndex:
                letter = char
                break
        inputText += letter        
        if letter == ' ' or letter == '\n':
            i += 1
      
    kerback.clear_session()
    del(NNWordOnlyModel
    return inputText

#inputText = '(Gdy od płaczącej matki, pod Twoją opiekę\n'
#sentenceLength  = 200
#fileName = 'Pan_Tadeusz'
#temp = 1.0
#seqLength = 25
#
#test = GenerateLetters(inputText, sentenceLength, temp, seqLength, fileName)
