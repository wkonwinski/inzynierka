import matplotlib.pyplot as plt
import pickle
import numpy as np

def DrawPlot(fileName, modelType):
    path = fileName + '_history' + modelType + '.txt'
    
    title = 'Tekst: '
    if fileName == 'Pan_Tadeusz':
        title += 'Pan Tadeusz     '
    elif fileName == 'Alicja':
        title += 'Alicja w krainie czarów     '
    elif fileName == 'news':
        title += 'Wiadomości     '
    elif fileName == 'Wesele':
        title += 'Wesele     '
      
    title += 'Model uczący się na: '
      
    if modelType == 'Letters':
        title += 'literach'
    elif modelType == 'WordsOnly':
        title += 'słowach ze znakami interpunkcyjnymi itd.'
    elif modelType == 'WaPWords':
        title += 'tylko słowach'
    elif modelType == 'historyWaPParts':
        title += 'częciach mowy'
    
    
    with open(path, "rb") as f:
        plotData = pickle.load(f)
        
    x = len(plotData['acc'])
    xAxes = np.linspace(0, x - 1, x)
        
    f, (firstSubplot, secoundSubplot) = plt.subplots(1, 2)
    
    firstSubplot.plot(xAxes, plotData['acc'], label = "Training Data Accuracy", color = 'navy')
    firstSubplot.plot(xAxes, plotData['val_acc'], label = "Validation Data Accuracy", color = 'crimson', linestyle = '-.')
    firstSubplot.set_title('Accuracy')
    firstSubplot.legend(loc = 'center right', borderaxespad = 0.5)
    firstSubplot.set_axisbelow(True)
    firstSubplot.minorticks_on()
    firstSubplot.grid(which='major', linestyle='-', linewidth='0.5', color='black')
    firstSubplot.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
    firstSubplot.set_xlim([0, x])
    plt.suptitle(title, fontsize = 16)
    
    secoundSubplot.plot(xAxes, plotData['loss'], label = "Training Data Loss", color = 'navy')
    secoundSubplot.plot(xAxes, plotData['val_loss'], label = "Validation Data Loss", color = 'crimson', linestyle = '-.')
    secoundSubplot.set_title('Cost Function Value')
    secoundSubplot.legend( loc = 'center right', borderaxespad = 0.5)
    secoundSubplot.set_axisbelow(True)
    secoundSubplot.minorticks_on()
    secoundSubplot.grid(which='major', linestyle='-', linewidth='0.5', color='black')
    secoundSubplot.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')
    secoundSubplot.set_xlim([0, x])
    
    f.show()

fileName = 'Pan_Tadeusz'
modelType = 'Letters'

DrawPlot(fileName, modelType)

