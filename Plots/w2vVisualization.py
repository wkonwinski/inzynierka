import numpy as np
import gensim
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def DrawTsne(modelWord2Vec, word, numberOfSimilarWords):    
    arr = np.empty((0, 100), dtype = 'f')
    wordLabels = [word]
    similarWords = modelWord2Vec.most_similar(positive=word, topn = numberOfSimilarWords)
    
    arr = np.append(arr, np.array([modelWord2Vec[word]]), axis = 0)
    for wordScore in similarWords:
        wordVec = modelWord2Vec[wordScore[0]]
        wordLabels.append(wordScore[0])
        arr = np.append(arr, np.array([wordVec]), axis = 0)
        
    tsne = TSNE(n_components = 2, random_state = 0)
    np.set_printoptions(suppress = True)
    Y = tsne.fit_transform(arr)

    x_coords = Y[:, 0]
    y_coords = Y[:, 1]

    plt.scatter(x_coords, y_coords)

    for label, x, y in zip(wordLabels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points', size = 20)
    plt.xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)
    plt.ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)
    plt.show()
    
path = r'../SharedData/'
modelWord2Vec = gensim.models.KeyedVectors.load_word2vec_format(path + 'nkjp+wiki-forms-all-100-skipg-hs.txt.gz')
word = 'piwo'
numberOfSimilarWords = 30

DrawTsne(modelWord2Vec, word, numberOfSimilarWords)

