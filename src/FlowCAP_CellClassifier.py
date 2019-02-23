#!/usr/bin/env python3
'''
This script will train a feed-forward neural network classifier to 25% of each
single subject in each data set of FlowCAP-I, and test the performance with the
remaining 75%. There are five data sets in FloWCAP-I: (1) NDD, (2) CFSE,
(3) StemCell, (4) Lymph, and (5) GvHD. The result of the mean and confidence
interval for each data set are
NDD ,  0.988706262232  ( 0.987414050849 0.989854916232 )
CFSE ,  0.999101182371  ( 0.998878315234 0.999307970954 )
StemCell ,  0.991351650942  ( 0.988360255002 0.993995231128 )
Lymph ,  0.98559997446  ( 0.976240110166 0.993195715049 )
GvHD ,  0.986439409178  ( 0.979937651836 0.991542823641 )


Created on Jul 30, 2016

@author: urishaham, huaminli
'''

import numpy as np
import os.path

from Util import DataHandler as dh
from Util import FileIO as io
from Util import feedforwadClassifier as net


'''
Parameters.

dataSet           - A list of names of FlowCAP-I data sets.
numSample         - A list of number of samples in each data set.
relevantMarkers   - A list of number of features in each data set.
hiddenLayersSizes - A list of 3 numbers to indicate the number of hidden
                    nodes in each hidden layer.
activation        - The activation function used to construct the classifier.
l2_penalty        - The regularization parameter to construct the classifier.
'''

dataSet = ['NDD', 'CFSE', 'StemCell', 'Lymph', 'GvHD']

numSample = [30, 13, 30, 30, 12]
relevantMarkers = [12, 8, 6, 5 ,6]

hiddenLayersSizes = [12, 6, 3]
activation = 'softplus'
l2_penalty = 1e-4

'''
The user needs to specify the data set to run the cell classifier.

Make your choice here - an integer from 0 to 4.
0: NDD
1: CFSE
2: StemCell
3: Lymph
4: GvHD
'''
choice = 4

# Generate the path of the chosen data set.
dataPath = os.path.join(io.DeepLearningRoot(), 'Data/FlowCAP-I/',
                        dataSet[choice])

# Generate the output table.
acc = np.zeros(numSample[choice])
F1 = np.zeros(numSample[choice])

'''
For each single sample of the chosen data set, train a feed-forward neural
net classifier using 25% of cells, and test the performance using the rest
75% of cells.
'''
print('Data set name: ', dataSet[choice])
for i in range(numSample[choice]):
    # Load sample.
    print('Load sample ', str(i+1))
    sample = dh.loadDeepCyTOFData(dataPath, i + 1,
                                  range(relevantMarkers[choice]), 'CSV',
                                  skip_header = 1)
    
    # Pre-process sample.
    print('Pre-process sample ', str(i+1))
    sample = dh.preProcessSamplesCyTOFData(sample)
    sample, preprocessor = dh.standard_scale(sample, preprocessor = None)
    
    # Split data into training and testing.
    print('Split data into training and testing.')
    trainSample, testSample = dh.splitData(sample, test_size = .75)
    
    # Train a feed-forward neural net classifier on the training data.
    print('Train a feed-forward neural net classifier on the training data.')
    classifier = net.trainClassifier(trainSample, dataSet[choice], i,
                                     hiddenLayersSizes,
                                     activation = activation,
                                     l2_penalty = l2_penalty)
    
    # Run the classifier on the testing data.
    print('Run the classifier on the testing data.')
    acc[i-1], F1[i-1], _ = net.prediction(testSample,
                                    dataSet[choice], i, classifier)
    
'''
Output the overall results.
'''
CI = np.zeros(10000)
for i in range(10000):
    CI[i] = np.mean(np.random.choice(F1, size = 30))
CI = np.sort(CI)
print(dataSet[choice], ', ', np.mean(CI), ' (', CI[250], CI[9750],')')

    