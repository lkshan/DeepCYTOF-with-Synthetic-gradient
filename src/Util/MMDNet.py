
from keras import callbacks as cb
from keras.callbacks import LearningRateScheduler
from keras import initializers
from keras.layers import Input, Dense, merge, Dropout, Activation, Add
from keras.layers.normalization import BatchNormalization
from keras.models import Model
import keras.optimizers as opt
from keras.regularizers import l2
import math
import matplotlib.pyplot as plt
import numpy as np
import os.path

from Util import CostFunctions as cf
from Util import Monitoring as mn
from Util import FileIO as io


class Sample:
    X = None
    y = None
    def __init__(self, X, y = None):
        self.X = X
        self.y = y

def step_decay(epoch):
    initial_lrate = 1e-5#1e-3
    drop = 0.1
    epochs_drop = 15.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

def constructMMD(target):

    mmdNetLayerSizes = [25, 25]
    l2_penalty = 1e-2
    init = lambda shape, name:initializers.normal(shape,
                                                     scale=.1e-4, name=name)
    space_dim = target.X.shape[1]
    
    calibInput = Input(shape=(space_dim,))
    block1_bn1 = BatchNormalization()(calibInput)
    block1_a1 = Activation('relu')(block1_bn1)
    block1_w1 = Dense(mmdNetLayerSizes[0], activation='linear',
                      W_regularizer=l2(l2_penalty), init = init)(block1_a1) 
    block1_bn2 = BatchNormalization()(block1_w1)
    block1_a2 = Activation('relu')(block1_bn2)
    block1_w2 = Dense(space_dim, activation='linear',
                      W_regularizer=l2(l2_penalty), init = init)(block1_a2) 
    block1_output = merge([block1_w2, calibInput], mode = 'sum')
    block2_bn1 = BatchNormalization()(block1_output)
    block2_a1 = Activation('relu')(block2_bn1)
    block2_w1 = Dense(mmdNetLayerSizes[1], activation='linear',
                      W_regularizer=l2(l2_penalty), init = init)(block2_a1) 
    block2_bn2 = BatchNormalization()(block2_w1)
    block2_a2 = Activation('relu')(block2_bn2)
    block2_w2 = Dense(space_dim, activation='linear',
                      W_regularizer=l2(l2_penalty), init = init)(block2_a2) 
    block2_output = merge([block2_w2, block1_output], mode = 'sum')
    block3_bn1 = BatchNormalization()(block2_output)
    block3_a1 = Activation('relu')(block3_bn1)
    block3_w1 = Dense(mmdNetLayerSizes[1], activation='linear',
                      W_regularizer=l2(l2_penalty), init = init)(block3_a1) 
    block3_bn2 = BatchNormalization()(block3_w1)
    block3_a2 = Activation('relu')(block3_bn2)
    block3_w2 = Dense(space_dim, activation='linear',
                      W_regularizer=l2(l2_penalty), init = init)(block3_a2) 
    block3_output = merge([block3_w2, block2_output], mode = 'sum')
    
    calibMMDNet = Model(input=calibInput, output=block3_output)
    

    return calibMMDNet, block3_output

def calibrate(target, source, sourceIndex, predLabel, path):
    
    mmdNetLayerSizes = [25, 25]
    l2_penalty = 1e-2
    #init = lambda shape:initializers.normal(shape, scale=.1e-4)
    space_dim = target.X.shape[1]
    
    calibInput = Input(shape=(space_dim,))
    block1_bn1 = BatchNormalization()(calibInput)
    block1_a1 = Activation('relu')(block1_bn1)
    block1_w1 = Dense(mmdNetLayerSizes[0], activation='linear',
                      W_regularizer=l2(l2_penalty), init = 'random_uniform')(block1_a1) 
    block1_bn2 = BatchNormalization()(block1_w1)
    block1_a2 = Activation('relu')(block1_bn2)
    block1_w2 = Dense(space_dim, activation='linear',
                      W_regularizer=l2(l2_penalty), init = 'random_uniform')(block1_a2) 
    block1_output = Add()([block1_w2, calibInput])
    block2_bn1 = BatchNormalization()(block1_output)
    block2_a1 = Activation('relu')(block2_bn1)
    block2_w1 = Dense(mmdNetLayerSizes[1], activation='linear',
                      W_regularizer=l2(l2_penalty), init = 'random_uniform')(block2_a1) 
    block2_bn2 = BatchNormalization()(block2_w1)
    block2_a2 = Activation('relu')(block2_bn2)
    block2_w2 = Dense(space_dim, activation='linear',
                      W_regularizer=l2(l2_penalty), init = 'random_uniform')(block2_a2) 
    block2_output = Add()([block2_w2, block1_output])
    block3_bn1 = BatchNormalization()(block2_output)
    block3_a1 = Activation('relu')(block3_bn1)
    block3_w1 = Dense(mmdNetLayerSizes[1], activation='linear',
                      W_regularizer=l2(l2_penalty), init = 'random_uniform')(block3_a1) 
    block3_bn2 = BatchNormalization()(block3_w1)
    block3_a2 = Activation('relu')(block3_bn2)
    block3_w2 = Dense(space_dim, activation='linear',
                      W_regularizer=l2(l2_penalty), init = 'random_uniform')(block3_a2) 
    block3_output = Add()([block3_w2, block2_output])
    
    calibMMDNet = Model(input=calibInput, output=block3_output)

    n = target.X.shape[0]
    p = np.random.permutation(n)
    toTake = p[range(int(.2*n))] 
    targetXMMD = target.X[toTake]
    targetYMMD = target.y[toTake]
    
    targetXMMD = targetXMMD[targetYMMD!=0]
    targetYMMD = targetYMMD[targetYMMD!=0]
    
    targetYMMD = np.reshape(targetYMMD, (-1, 1))

    n = source.X.shape[0]
    p = np.random.permutation(n)
    toTake = p[range(int(.2*n))] 
    sourceXMMD = source.X[toTake]
    sourceYMMD = predLabel[toTake]
    
    sourceXMMD = sourceXMMD[sourceYMMD!=0]
    sourceYMMD = sourceYMMD[sourceYMMD!=0]
    
    sourceYMMD = np.reshape(sourceYMMD, (-1, 1))

    lrate = LearningRateScheduler(step_decay)
    optimizer = opt.rmsprop(lr=0.0)
    calibMMDNet.compile(optimizer = optimizer, loss = lambda y_true,y_pred: 
       cf.MMD(block3_output, targetXMMD, 
            MMDTargetValidation_split = 0.1).KerasCost(y_true,y_pred))

    sourceLabels = np.zeros(sourceXMMD.shape[0])

    calibMMDNet.fit(sourceXMMD,sourceLabels,nb_epoch=500,
            batch_size=1000,validation_split=0.1,verbose=0,
            callbacks=[lrate,mn.monitorMMD(sourceXMMD, sourceYMMD, targetXMMD,
                                           targetYMMD, calibMMDNet.predict),
              cb.EarlyStopping(monitor='val_loss',patience=20,mode='auto')])
    plt.close('all')
    calibMMDNet.save_weights(os.path.join(io.DeepLearningRoot(),
                                          'savemodels/' + path + '/ResNet'+ str(sourceIndex)+'.h5'))
    calibrateSource = Sample(calibMMDNet.predict(source.X),
                                             source.y)
    calibMMDNet = None
    return calibrateSource

def loadModel(target, source, sourceIndex, predLabel, path):
    mmdNetLayerSizes = [25, 25]
    l2_penalty = 1e-2
    init = lambda shape, name:initializers.normal(shape,
                                                     scale=.1e-4, name=name)
    space_dim = target.X.shape[1]
    
    calibInput = Input(shape=(space_dim,))
    block1_bn1 = BatchNormalization()(calibInput)
    block1_a1 = Activation('relu')(block1_bn1)
    block1_w1 = Dense(mmdNetLayerSizes[0], activation='linear',
                      W_regularizer=l2(l2_penalty), init = init)(block1_a1) 
    block1_bn2 = BatchNormalization()(block1_w1)
    block1_a2 = Activation('relu')(block1_bn2)
    block1_w2 = Dense(space_dim, activation='linear',
                      W_regularizer=l2(l2_penalty), init = init)(block1_a2) 
    block1_output = merge([block1_w2, calibInput], mode = 'sum')
    block2_bn1 = BatchNormalization()(block1_output)
    block2_a1 = Activation('relu')(block2_bn1)
    block2_w1 = Dense(mmdNetLayerSizes[1], activation='linear',
                      W_regularizer=l2(l2_penalty), init = init)(block2_a1) 
    block2_bn2 = BatchNormalization()(block2_w1)
    block2_a2 = Activation('relu')(block2_bn2)
    block2_w2 = Dense(space_dim, activation='linear',
                      W_regularizer=l2(l2_penalty), init = init)(block2_a2) 
    block2_output = merge([block2_w2, block1_output], mode = 'sum')
    block3_bn1 = BatchNormalization()(block2_output)
    block3_a1 = Activation('relu')(block3_bn1)
    block3_w1 = Dense(mmdNetLayerSizes[1], activation='linear',
                      W_regularizer=l2(l2_penalty), init = init)(block3_a1) 
    block3_bn2 = BatchNormalization()(block3_w1)
    block3_a2 = Activation('relu')(block3_bn2)
    block3_w2 = Dense(space_dim, activation='linear',
                      W_regularizer=l2(l2_penalty), init = init)(block3_a2) 
    block3_output = merge([block3_w2, block2_output], mode = 'sum')
    
    calibMMDNet = Model(input=calibInput, output=block3_output)

    calibMMDNet.load_weights(os.path.join(io.DeepLearningRoot(),
                                          'savemodels/'+ path + '/ResNet'+ str(sourceIndex)+'.h5'))
    
    return calibMMDNet