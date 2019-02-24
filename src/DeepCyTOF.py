#!/usr/bin/env python3
'''
This script will train a DeepCyTOF neural network classifier on three CyTOF
datasets. For each dataset, DeepCyTOF will (1) pick a reference sample and
train a feed-forward neural network classifier on the reference sample;
(2) train a MMD-ResNet for each pair of the remaining sample and the reference
sample; (3) predict the cell label information for each remaining sample.

Created on Jul 30, 2016
Author: urishaham, huaminli

Updated by Lukas Hanincik
Updated on 24.02.2019
'''
from keras import backend as K
import numpy as np
import os.path

from Util import CostFunctions as cf
from Util import DataHandler as dh
from Util import denoisingAutoEncoder as dae
from Util import FileIO as io
from Util import feedforwadClassifier as net
from Util import MMDNet as mmd

from Util.MMDNet_SG import ModelSG


class Sample:
    X = None
    y = None
    def __init__(self, X, y = None):
        self.X = X
        self.y = y
        
'''
Parameters.

dataSet           - A list of names of CyTOF data sets.
isCalibrate       - A indicator whether perform calibration or not.
denoise           - A indicator whether run de-noising auto-encoder or not.
loadModel         - A indicator whether load previous trained model or not.
hiddenLayersSizes - A list of 3 numbers to indicate the number of hidden
                    nodes in each hidden layer.
activation        - The activation function used to construct the classifier.
l2_penalty        - The regularization parameter to construct the classifier.
keepProb          - The keep probability for each single cell to be used as
                    the training set of de-noising autoencoder.
'''
dataSet = ['MultiCenter_16sample',
           '56_unnormalized', '56_normalized',
           'old&young_unnomral', 'old&young_normal']

isCalibrate = True
denoise = True
loadModel = True

hiddenLayersSizes = [12,6,3]
activation = 'softplus'
l2_penalty = 1e-4

'''
The user needs to specify the data set to run DeepCyTOF.

Make your choice here - an integer from 0 to 4.
0: MultiCenter study
1: 56 unnormalized
2: 56 normalized
3: 134 unnormalized
4: 134 normalized
'''
choice = 0
#dataPath = '/home/hl475/Documents/deepcytof/Data/' + dataSet[choice]
dataPath = os.path.join(os.path.dirname(os.path.realpath(__file__)),'..', 'Data', dataSet[choice])
if choice == 0:
    dataIndex = np.arange(1,16+1)
    trainIndex = dataIndex
    testIndex = dataIndex
    relevantMarkers = np.asarray([1,2,3,4,5,6,7,8])-1
    mode = 'CSV'
    numClasses = 4
    keepProb = .8
elif choice == 1 or choice == 2:
    testIndex = np.arange(1,56+1)
    trainIndex = testIndex[0:56:4]
    relevantMarkers = np.asarray([3,9,10,13,15,28,34,39,40,43,44,45])-1
    if choice == 1:
        mode = 'FCS'
    else:
        mode = 'CSV'
    numClasses = 5
    keepProb = .9
elif choice == 3 or choice == 4:
    testIndex = np.arange(1,136+1)
    trainIndex = testIndex[0:136:4]
    relevantMarkers = np.asarray([3,9,10,13,15,28,34,39,40,43,44,45])-1
    if choice == 1:
        mode = 'FCS'
    else:
        mode = 'CSV'
    numClasses = 5
    keepProb = .9

'''
Choose the reference sample.
'''
print('Choose the reference sample between ' + str(trainIndex))
refSampleInd = dh.chooseReferenceSample(dataPath, trainIndex,
                                        relevantMarkers, mode,
                                        choice)

print('Load the target ' + str(trainIndex[refSampleInd])) 
target = dh.loadDeepCyTOFData(dataPath, trainIndex[refSampleInd],
                              relevantMarkers, mode)

# Pre-process sample.
target = dh.preProcessSamplesCyTOFData(target)

'''
Train the de-noising auto encoder.
'''
print('Train the de-noising auto encoder.')
DAE = dae.trainDAE(target, dataPath, refSampleInd, trainIndex,
                             relevantMarkers, mode, keepProb, denoise,
                             loadModel, dataSet[choice])
denoiseTarget = dae.predictDAE(target, DAE, denoise)

'''
Train the feed-forward classifier on (de-noised) target.
'''
denoiseTarget, preprocessor = dh.standard_scale(denoiseTarget,
                                                preprocessor = None)

if loadModel:
    from keras.models import load_model
    cellClassifier = load_model(os.path.join(io.DeepLearningRoot(),
                                'savemodels/' + dataSet[choice] +
                                '/cellClassifier.h5'))
else:
    print('Train the classifier on de-noised Target')
    cellClassifier = net.trainClassifier(denoiseTarget, mode, refSampleInd,
                                         hiddenLayersSizes,
                                         activation,
                                         l2_penalty,
                                         dataSet[choice])
    
'''
Test the performance with and without calibration.
'''
# Generate the output table.
dim = 2 if isCalibrate else 1    
acc = np.zeros((testIndex.size, dim), np.float16)
F1 = np.zeros((testIndex.size, dim), np.float16)
mmd_before = np.zeros(testIndex.size)
mmd_after = np.zeros(testIndex.size)

for i in np.arange(testIndex.size):
    # Load the source.
    sourceIndex = testIndex[i]
    source = dh.loadDeepCyTOFData(dataPath, sourceIndex, relevantMarkers, mode)
    source = dh.preProcessSamplesCyTOFData(source)
    
    # De-noising the source.
    denoiseSource = dae.predictDAE(source, DAE, denoise)
    denoiseSource, _ = dh.standard_scale(denoiseSource,
                                         preprocessor = preprocessor)
    
    # Predict the cell type of the source.
    print('Run the classifier on source ', str(sourceIndex),
          'without calibration')
    acc[i,0], F1[i,0], predLabel = net.prediction(denoiseSource,
                                                mode, i,
                                               cellClassifier)
    sourceInds = np.random.randint(low=0, high = source.X.shape[0], size = 1000)
    targetInds = np.random.randint(low=0, high = target.X.shape[0], size = 1000)
    mmd_before[i] = K.eval(cf.MMD(denoiseSource.X, denoiseTarget.X).cost(
        K.variable(value=denoiseSource.X[sourceInds]),
        K.variable(value=denoiseTarget.X[targetInds])))
    print('MMD before: ', str(mmd_before[i]))
    if isCalibrate:
        loadModel = False
        if loadModel:
            calibMMDNet = mmd.loadModel(denoiseTarget, denoiseSource,
                                            sourceIndex, predLabel, dataSet[choice])
            calibrateSource = Sample(calibMMDNet.predict(denoiseSource.X),
                                            denoiseSource.y)
            calibMMDNet = None
        else:
            """
            calibrateSource = mmd.calibrate(denoiseTarget, denoiseSource,
                                            sourceIndex, predLabel, dataSet[choice])
            """

            calibrateSource = ModelSG(denoiseTarget, denoiseSource,
                                            sourceIndex, predLabel, dataSet[choice])
            
        print('Run the classifier on source ', str(sourceIndex),
          'with calibration')
        acc[i,1], F1[i,1], predLabell = net.prediction(calibrateSource,
                                                mode, i,
                                           cellClassifier)
        mmd_after[i] = K.eval(cf.MMD(calibrateSource.X, denoiseTarget.X).cost(
            K.variable(value=calibrateSource.X[sourceInds]),
            K.variable(value=denoiseTarget.X[targetInds])))
        print('MMD after: ', str(mmd_after[i]))
        calibrateSource = None
    source = None
    denoiseSource = None
    
'''
Output the overall results.
'''
CI = np.zeros(10000)
for i in range(10000):
    CI[i] = np.mean(np.random.choice(F1[:,0], size = 30))
CI = np.sort(CI)
print(mode, ', ', np.mean(CI), ' (', CI[250], CI[9750],')')

if isCalibrate:
    CI = np.zeros(10000)
    for i in range(10000):
        CI[i] = np.mean(np.random.choice(F1[:,1], size = 30))
    CI = np.sort(CI)
    print(mode, ', ', np.mean(CI), ' (', CI[250], CI[9750],')')