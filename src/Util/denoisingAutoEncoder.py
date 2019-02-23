from keras import callbacks as cb
from keras.layers import Input, Dense, merge, Dropout, Activation
from keras.models import Model
from keras.regularizers import l2
import matplotlib.pyplot as plt
import numpy as np

from Util import Monitoring as mn
from Util import DataHandler as dh
import os.path
from Util import FileIO as io

class Sample:
    X = None
    y = None
    def __init__(self, X, y = None):
        self.X = X
        self.y = y
        
def trainDAE(target, dataPath, refSampleInd, trainIndex, relevantMarkers, mode,
             keepProb, denoise, loadModel, path):
    sourceX = []
    for i in np.arange(trainIndex.size-1):
        sourceIndex = np.delete(trainIndex, refSampleInd)[i]
        source = dh.loadDeepCyTOFData(dataPath, sourceIndex,
                                      relevantMarkers, mode)
        numZerosOK=1
        toKeepS = np.sum((source.X==0), axis = 1) <= numZerosOK
        if i == 0:
            sourceX = source.X[toKeepS]
        else:
            sourceX = np.concatenate([sourceX, source.X[toKeepS]], axis = 0)
        
    # preProcess source
    sourceX = np.log(1 + np.abs(sourceX))
    
    numZerosOK=1
    toKeepT = np.sum((target.X==0), axis = 1) <= numZerosOK
    
    inputDim = target.X.shape[1]
    
    ae_encodingDim = 25
    l2_penalty_ae = 1e-2
    
    if denoise:
        if loadModel:
            from keras.models import load_model
            autoencoder = load_model(os.path.join(io.DeepLearningRoot(),
                                                  'savemodels/' + path + '/denoisedAE.h5'))
        else:
            # train de-noising auto encoder and save it.
            trainTarget_ae = np.concatenate([sourceX, target.X[toKeepT]],
                                            axis=0)
            trainData_ae = trainTarget_ae * np.random.binomial(n=1, p=keepProb,
                                                size = trainTarget_ae.shape)
        
            input_cell = Input(shape=(inputDim,))
            encoded = Dense(ae_encodingDim, activation='relu',
                            W_regularizer=l2(l2_penalty_ae))(input_cell)
            encoded1 = Dense(ae_encodingDim, activation='relu',
                             W_regularizer=l2(l2_penalty_ae))(encoded)
            decoded = Dense(inputDim, activation='linear',
                            W_regularizer=l2(l2_penalty_ae))(encoded1)
        
            autoencoder = Model(input=input_cell, output=decoded)
            autoencoder.compile(optimizer='rmsprop', loss='mse')
            autoencoder.fit(trainData_ae, trainTarget_ae, nb_epoch=80,
                            batch_size=128, shuffle=True,
                            validation_split=0.1, verbose = 0,
                            callbacks=[mn.monitor(), cb.EarlyStopping(
                            monitor='val_loss', patience=25,  mode='auto')])
            autoencoder.save(os.path.join(io.DeepLearningRoot(),
                                          'savemodels/' + path + '/denoisedAE.h5'))
            del sourceX
            plt.close('all')
        
        return autoencoder

def predictDAE(target, autoencoder, denoise = False):
    if denoise:    
        # apply de-noising auto encoder to target.
        denoiseTarget = Sample(autoencoder.predict(target.X), target.y)
    else:
        denoiseTarget = Sample(target.X, target.y)
        
    return denoiseTarget