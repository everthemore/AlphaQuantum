from __future__ import division
import numpy as np
import tensorflow as tf
from keras.models import Sequential, load_model, Model
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, Activation, LeakyReLU, add
from keras.optimizers import SGD
from keras import regularizers
import keras.backend as K


from enumerate_actions import *
import os

def softmax_cross_entropy_with_logits(y_true, y_pred):
	p = y_pred
	pi = y_true

	zero = tf.zeros(shape = tf.shape(pi), dtype=tf.float32)
	where = tf.equal(pi, zero)

	negatives = tf.fill(tf.shape(pi), -100.0)
	p = tf.where(where, negatives, p)

	loss = tf.nn.softmax_cross_entropy_with_logits(labels = pi, logits = p)

	return loss


class Network():
    def __init__(self, config):
        """
        A wrapper around a residual neural network tower that is used to predict
        new moves and outcomes of a given (set of) game boards.

        Inputs
        ------
        config          - a valid Json configuration file
        """
        # Store the configuration arguments
        self.hidden_layers      = config["Network"]["hidden_layers"]
        self.num_layers         = len(self.hidden_layers)

        self.epochs             = config["Network"]["numEpochs"]
        self.batch_size         = config["Network"]["batchSize"]

        # Store some information about the game we're training our brain on
        self.board_x, self.board_y = 8, 8
        self.action_size           = config["action_size"]

        self.momentum           = config["Network"]["momentum"]
        self.reg_const          = config["Network"]["reg_const"]
        self.dropout            = config["Network"]["dropout"]
        self.learning_rate      = config["Network"]["learning_rate"]
        self.input_dim          = (8,8,105)

        self.model = self._build_model()

    def _build_model(self):
        main_input = Input(shape = self.input_dim, name = 'main_input')

        x = self.conv_layer(main_input, self.hidden_layers[0]['filters'], self.hidden_layers[0]['kernel_size'])

        if len(self.hidden_layers) > 1:
           for h in self.hidden_layers[1:]:
               x = self.residual_layer(x, h['filters'], h['kernel_size'])

        vh = self.value_head(x)
        ph = self.policy_head(x)

        model = Model(inputs=[main_input], outputs=[vh, ph])
        model.compile(loss={'value_head': 'mean_squared_error', 'policy_head': softmax_cross_entropy_with_logits},
            optimizer=SGD(lr=self.learning_rate, momentum = self.momentum),
            loss_weights={'value_head': 0.5, 'policy_head': 0.5}
            )

        model.summary()
        return model

    def train(self, examples):
        """
        Train the network using the given examples.

        Inputs
        ------
        examples (list): list of examples, each example is of form (board, pi, v)
        """
        input_boards, target_pis, target_vs = list(zip(*examples))
        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        return self.model.fit(x = input_boards, y = [target_pis, target_vs], batch_size = self.batch_size, epochs = self.epochs, verbose=0)
#        return self.model.fit(states, targets, epochs=epochs, verbose=verbose, validation_split = validation_split, batch_size = batch_size)

    def predict(self, boards):
        """
        Predict the policy vector and value for a given board.

        Inputs
        ------
        board: np array representing the game board
        """
        boards = boards[np.newaxis, :, :, :]
        v, pi = self.model.predict(boards)
        return pi[0], v[0][0]

    def value_head(self, x):
        x = Conv2D(
        		filters = 1
        		, kernel_size = (1,1)
        		, data_format="channels_last"
        		, padding = 'same'
        		, use_bias=False
        		, activation='linear'
        		, kernel_regularizer = regularizers.l2(self.reg_const)
        )(x)

        #x = BatchNormalization(axis=1)(x)
        x = LeakyReLU()(x)
        x = Flatten()(x)

        x = Dense(
        	20
        	, use_bias=False
        	, activation='linear'
        	, kernel_regularizer=regularizers.l2(self.reg_const)
        	)(x)

        x = LeakyReLU()(x)

        x = Dense(
        	1
        	, use_bias=False
        	, activation='tanh'
        	, kernel_regularizer=regularizers.l2(self.reg_const)
        	, name = 'value_head'
        	)(x)

        return (x)

    def policy_head(self, x):
        x = Conv2D(
        	filters = 2
        	, kernel_size = (1,1)
        	, data_format="channels_last"
        	, padding = 'same'
        	, use_bias=False
        	, activation='linear'
        	, kernel_regularizer = regularizers.l2(self.reg_const)
        	)(x)

        #x = BatchNormalization(axis=1)(x)
        x = LeakyReLU()(x)
        x = Flatten()(x)

        x = Dense(
        	self.action_size
        	, use_bias=False
        	, activation='softmax'
        	, kernel_regularizer=regularizers.l2(self.reg_const)
        	, name = 'policy_head'
        	)(x)

        return (x)

    def residual_layer(self, input_block, filters, kernel_size):

        x = self.conv_layer(input_block, filters, kernel_size)

        x = Conv2D(
        filters = filters
        , kernel_size = kernel_size
        , data_format="channels_last"
        , padding = 'same'
        , use_bias=False
        , activation='linear'
        , kernel_regularizer = regularizers.l2(self.reg_const)
        )(x)

        #x = BatchNormalization(axis=1)(x)
        x = add([input_block, x])
        x = LeakyReLU()(x)

        return (x)

    def conv_layer(self, x, filters, kernel_size):

        x = Conv2D(
        filters = filters
        , kernel_size = kernel_size
        , data_format="channels_last"
        , padding = 'same'
        , use_bias=False
        , activation='linear'
        , kernel_regularizer = regularizers.l2(self.reg_const)
        )(x)

        #x = BatchNormalization(axis=1)(x)
        x = LeakyReLU()(x)

        return (x)

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        self.model.save_weights(filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            print("No model in path {}".format(filepath))
            return 0
        self.model.load_weights(filepath)

# def __buildnet(self):
#     """
#     Build the network model using Keras.
#     """
#     # Neural Net
#     self.input_boards = Input(shape=(self.len_history, self.board_x, self.board_y, self.num_planes))
#     x_image = Reshape((self.len_history, self.board_x, self.board_y, self.num_planes))(self.input_boards)
#
#     np.random.seed(0)
#     h_conv1 = Activation('relu')(BatchNormalization()(Conv2D(self.numFilters, self.convsize, padding='same',data_format='channels_first')(x_image)))
#     h_conv2 = Activation('relu')(BatchNormalization()(Conv2D(self.numFilters, self.convsize, padding='same',data_format='channels_first')(h_conv1)))
#     h_conv3 = Activation('relu')(BatchNormalization()(Conv2D(self.numFilters, 3, padding='valid',data_format='channels_first')(h_conv2)))
#     h_conv4 = Activation('relu')(BatchNormalization()(Conv2D(self.numFilters, 3, padding='valid',data_format='channels_first')(h_conv3)))
#
#     h_conv4_flat = Flatten()(h_conv4)
#     s_fc1 = Dropout(self.dropout)(Activation('sigmoid')(BatchNormalization()(Dense(64)(h_conv4_flat))))
#     s_fc2 = Dropout(self.dropout)(Activation('sigmoid')(BatchNormalization()(Dense(64)(s_fc1))))
#     self.pi = Dense(self.action_size, activation='softmax', name='pi')(s_fc2)
#     self.v = Dense(1, activation='tanh')(Dense(1, activation='linear', name='v')(s_fc2))
#
#     # Assemble the model
#     self.model = Model(inputs=self.input_boards, outputs=[self.pi, self.v])
#     self.model.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=Adam(self.lr, clipnorm=1))
