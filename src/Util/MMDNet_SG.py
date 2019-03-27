import random
import tensorflow as tf
import numpy as np

from tqdm import tqdm

from keras import backend as K
from keras.layers import Dense, BatchNormalization, Activation, Input, Add
from keras.regularizers import l2
from keras.models import Model
from keras import optimizers as opt

from tensorflow.train import RMSPropOptimizer

from Util import CostFunctions as cf
IntType = 'int32'

class Sample:
    X = None
    y = None
    def __init__(self, X, y = None):
        self.X = X
        self.y = y

class Layer(object):
    """ Layer class for creating dense layers.

    This class Keras layers as its backbone to create
    dense layers with BatchNorm and relu activation.

    Attributes:
        units: An integer number of nodes of a layer.
        inputs: A tensor that will be input to the layer.
        name: A string for defintion and tf.scope.
        out: A boolean indicating whether to add BatchNorm 
            and relu activation.
        sg: A boolean indicating whether the layer is for
            synthetic gradients.
    """

    """
    def __init__(self, units, inputs, name, out=False, sg=False):
        self.name = name
        with tf.variable_scope(self.name):
            if sg:
                inputs_c = K.concatenate(inputs, 1)
                self.output = Dense(units, kernel_initializer=tf.zeros_initializer())(inputs_c)
            else:
                self.output = Dense(units)(inputs)
                if not out: self.output = Activation('relu')(BatchNormalization()(self.output))
        self.layer_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=self.name)
    """

    def __init__(self, units, inputs, name, l2_penalty=None, sg=False, out=False):
        self.name = name
        with tf.variable_scope(self.name):
            if sg:
                inputs_c = K.concatenate(inputs, 1)
                self.output = Dense(units, kernel_initializer=tf.zeros_initializer())(inputs_c)
            else:
                if not out:
                    self.output = BatchNormalization()(inputs)
                    self.output = Activation('relu')(self.output)
                else: self.output = inputs
                self.output = Dense(units, activation='linear',
                      W_regularizer=l2(l2_penalty), init = 'random_uniform')(self.output)
        self.layer_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=self.name)

class ModelSG(object):
    """Model for demonstraing synthetic gradients.

    A model that uses Keras and Tensorflow to 
    demonstrate decoupled training through synthetic 
    gradients on MNIST dataset.

    Attributes:
        sess: A tf.sess() that will be used by the model.
    """
    
    def __init__(self, target, source, sourceIndex, predLabel, path, sg_only=True):
        sess = tf.Session()
        K.set_session(sess)
        
        self.sess = sess
        self.target = target
        self.source = source
        self.sourceIndex = sourceIndex
        self.predLabel = predLabel
        self.path = path
        self.sg_only = sg_only

        self.lr_div = 10
        self.lr_div_steps = 50
        self.l2_penalty = 1e-2
        self.itterations = 2500
        self.batch_size = 1000
        self.sg_pp = .2
        self.init_lr = 3e-5
        
        self.testingData = []

        self.create_layers()
        
        self.train(self.itterations, self.batch_size, self.sg_pp, self.init_lr) # number of epochs, batch size, update probability, learning rate
        # try lerning rate 1e-5, 1e-3
        
        self.calibratedData = Sample(self.finalCalibration(), self.source.y)

    def create_layers(self):
        """Creates normal and synthetic layers for the graph
        """
        mmdNetLayerSizes = [25, 25]
        #init = lambda shape:initializers.normal(shape, scale=.1e-4)
        space_dim = self.target.X.shape[1]
        
        # Inputs
        X = tf.placeholder(tf.float32, shape=(None, self.source.X.shape[1]), name="data")
        Y = tf.placeholder(tf.float32, shape=(None, self.source.X.shape[1]), name="labels")
        self.inputs = [X,Y]
        
        calibInput = Input(shape=(space_dim,), tensor=X)
        
        block1 = Layer(mmdNetLayerSizes[0], calibInput, 'block1', self.l2_penalty)
        
        block2 = Layer(space_dim, block1.output, 'block2', self.l2_penalty)
        
        with tf.variable_scope('block2_addition'):
            block2_output = Add()([block2.output, calibInput])
        
        block3 = Layer(mmdNetLayerSizes[1], block2_output, 'block3', self.l2_penalty)
        
        block4 = Layer(space_dim, block3.output, 'block4', self.l2_penalty)
        
        with tf.variable_scope('block4_addition'):
            block4_output = Add()([block4.output, block2_output])

        block5 = Layer(mmdNetLayerSizes[1], block4_output, 'block5', self.l2_penalty)
        
        block6 = Layer(space_dim, block5.output, 'block6', self.l2_penalty)
        
        with tf.variable_scope('block6_addition'):
            block6_output = Add()([block6.output, block4_output])
            
        logits = Layer(space_dim, block6_output, 'block7', self.l2_penalty, out=True)

        self.layers = [block1, block2, block3, block4, block5, block6, logits]

        # sg layers
        synth_b1 = Layer(mmdNetLayerSizes[0], [block1.output,Y], 'sg2', sg=True)
        synth_b2 = Layer(space_dim, [block2.output,Y], 'sg3', sg=True)
        synth_b3 = Layer(mmdNetLayerSizes[1], [block3.output,Y], 'sg4', sg=True)
        synth_b4 = Layer(space_dim, [block4.output,Y], 'sg5', sg=True)
        synth_b5 = Layer(mmdNetLayerSizes[1], [block5.output,Y], 'sg6', sg=True)
        synth_b6 = Layer(space_dim, [block6.output,Y], 'sg7', sg=True)
        
        self.synth_layers = [synth_b1, synth_b2, synth_b3, synth_b4, synth_b5, synth_b6]
    
    def train_layer_n(self, h_m, h_n, d_hat_m, class_loss, next_l, d_n=None, p=True):
        """Creates computation graphs for decoupled training through 
            applying synthetic gradients on gradients of the layers.
        
            Args:
                h_m: An integer index of a layer that will be updated.
                h_n: An integer index of a layer from which grads of `h_m` will be calculated.
                d_hat_m: An integer index of a synthetic layer that accompanies `h_m`.
                class_loss: A tensor that contains the prediction loss.
                next_l: An integer index of a following layer of `h_m`.
                d_n: An integer index of a synthetic layer whose grads will be applied to grads of `h_m`.
                p: A boolean indicating whether 
        """
        if d_n is not None: d_n = self.synth_layers[d_n].output
        if p: h_n = self.layers[h_n].output
        with tf.variable_scope(self.layers[h_m].name):
            layer_grads = tf.gradients(h_n, [self.layers[h_m].output]+self.layers[next_l].layer_vars, d_n)
            layer_gv = list(zip(layer_grads[1:],self.layers[next_l].layer_vars))
            layer_opt = RMSPropOptimizer(learning_rate=self.learning_rate).apply_gradients(layer_gv)
        with tf.variable_scope(self.synth_layers[d_hat_m].name):
            d_m = layer_grads[0]
            sg_loss = tf.divide(tf.losses.mean_squared_error(self.synth_layers[d_hat_m].output, d_m), class_loss)
            sg_opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(sg_loss, var_list=self.synth_layers[d_hat_m].layer_vars)
        return layer_opt, sg_opt
    
    def prepare_training(self, learning_rate, targetXMMD):
        """Creates necessary computation graphs for training.

        Args:
            learning_rate: A float indicating the initial learning rate.
        """
        self.learning_rate = tf.Variable(learning_rate, dtype=tf.float32, name="lr")
        self.reduce_lr = tf.assign(self.learning_rate, self.learning_rate/self.lr_div, name="lr_decrease")

        #self.pred_loss = tf.losses.cosine_distance(labels=targetXMMD, predictions=self.layers[6].output, scope="prediction_loss", axis=1, reduction=tf.losses.Reduction.MEAN)
        #self.pred_loss = tf.losses.softmax_cross_entropy(onehot_labels=self.inputs[1], logits=self.layers[6].output, scope="prediction_loss")
        self.pred_loss = cf.MMD(self.layers[6].output, targetXMMD, MMDTargetValidation_split = 0.1).KerasCost([],[])
        
        block7_opt, sg7_opt = self.train_layer_n(5, self.pred_loss, 5, self.pred_loss, 6, p=False)
        block6_opt, sg6_opt = self.train_layer_n(4, 5, 4, self.pred_loss, 5, 5)
        block5_opt, sg5_opt = self.train_layer_n(3, 4, 3, self.pred_loss, 4, 4)
        block4_opt, sg4_opt = self.train_layer_n(2, 3, 2, self.pred_loss, 3, 3)
        block3_opt, sg3_opt = self.train_layer_n(1, 2, 1, self.pred_loss, 2, 2)
        block2_opt, sg2_opt = self.train_layer_n(0, 1, 0, self.pred_loss, 1, 1)
        
        block1_opt = RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.layers[0].output, var_list=self.layers[0].layer_vars, 
                                            grad_loss=self.synth_layers[0].output)

        self.decoupled_training = [[block1_opt],
                                   [block2_opt, sg2_opt],
                                   [block3_opt, sg3_opt],
                                   [block4_opt, sg4_opt],
                                   [block5_opt, sg5_opt],
                                   [block6_opt, sg6_opt],
                                   [block7_opt, sg7_opt]]

        self.bpropOptimizer = RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.pred_loss)

    def train(self, iterations, batch_size, update_prob, learning_rate):
        """Trains the model in a decoupled way on a MNIST dataset.
        
        Args:
            iterations: An integer for how many iterations the model will train.
            batch_size: An integer for the size of the batches.
            update_prob: A float indicating how often layers should be
                updated in a decoupled fashion.
            learning_rate: A float indicating the initial learning rate.
        """
        
        n = self.target.X.shape[0]
        p = np.random.permutation(n)
        toTake = p[range(int(.2*n))]
        targetXMMD = self.target.X[toTake]
        targetYMMD = self.target.y[toTake]
        
        targetXMMD = targetXMMD[targetYMMD!=0]
        targetYMMD = targetYMMD[targetYMMD!=0]
        
        targetYMMD = np.reshape(targetYMMD, (-1, 1))
    
        n = self.source.X.shape[0]
        p = np.random.permutation(n)
        toTake = p[range(int(.2*n))] 
        sourceXMMD = self.source.X[toTake]
        sourceYMMD = self.predLabel[toTake]
        
        sourceXMMD = sourceXMMD[sourceYMMD!=0]
        sourceYMMD = sourceYMMD[sourceYMMD!=0]
        
        sourceYMMD = np.reshape(sourceYMMD, (-1, 1))
        
        sourceLabels = np.zeros(sourceXMMD.shape)
        
        self.prepare_training(learning_rate, targetXMMD)
        
        with self.sess.as_default():
            init = tf.global_variables_initializer()
            self.sess.run(init)
            testingData = self.test(batch_size, targetXMMD)
            self.testingData.append({'itteration':0, 'MMD': testingData})
            print('\n')
            print('Initial MMD: ', testingData)
            for i in tqdm(range(1,iterations+1)):
                if i % self.lr_div_steps == 0:
                    self.sess.run(self.reduce_lr)
                
                batch_indices = K.cast(K.round(K.random_uniform(shape=tuple([batch_size]), minval=0, 
                                                 maxval=sourceXMMD.shape[0]-1)),IntType)
                batchX = sourceXMMD[K.eval(batch_indices)]
                batchY = sourceLabels[K.eval(batch_indices)]
                
                X,Y = self.inputs[0], self.inputs[1]
                
                if self.sg_only :
                    for d in self.decoupled_training: 
                        if random.random() <= update_prob or True: self.sess.run(d, feed_dict={X:batchX,Y:batchY})
                else:
                    self.sess.run(self.bpropOptimizer, feed_dict={X:batchX,Y:batchY})
                    
                    
                if i % 50 == 0:
                    testingData = self.test(batch_size, targetXMMD)
                    self.testingData.append({'itteration':i, 'MMD': testingData})
                    print('\n')
                    print('MMD after ' + str(i) + ': ' + str(testingData))
    
    def test(self, batch_size, targetXMMD):
        """Tests the model on MNIST.test dataset

        Args:
            batch_size: An integer for the size of the batches.
        """
        X,Y = self.inputs[0], self.inputs[1]
        preds = self.sess.run(self.layers[6].output, feed_dict={X: self.source.X})
        
        final_mmd = K.eval(cf.MMD(preds, self.target.X).cost(
            K.variable(value=preds[np.random.randint(low=0, high=self.source.X.shape[0], size = batch_size)]),
            K.variable(value=self.target.X[np.random.randint(low=0, high=self.target.X.shape[0], size = batch_size)])))

        return final_mmd
    
    def finalCalibration(self):
        X,Y = self.inputs[0], self.inputs[1]
        return self.sess.run(self.layers[6].output, feed_dict={X: self.source.X})
# -*- coding: utf-8 -*-

