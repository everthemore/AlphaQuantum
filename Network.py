from __future__ import division
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
import os

class Network():
    def __init__(self, env, config):
        """
        A wrapper around a residual neural network tower that is used to predict
        new moves and outcomes of a given (set of) game boards.

        Inputs
        ------
        env             - an instance of an openAI Gym environment
        config          - a valid Json configuration file
        """
        # Store the configuration arguments
        self.numFilters         = config["Network"]["numFilters"]
        self.dropout            = config["Network"]["dropout"]
        self.convsize           = config["Network"]["kernelSize"]
        self.lr                 = config["Network"]["learnRate"]
        self.epochs             = config["Network"]["numEpochs"]
        self.batch_size         = config["Network"]["batchSize"]
        self.len_history        = config["numHistory"]

        # Store some information about the game we're training our brain on
        self.board_x, self.board_y = 2*env.board_size, 2*env.board_size
        self.action_size           = env.action_space.shape

        # Build the neural network
        self.__buildnet()

    def __buildnet(self):
        """
        Build the network model using Keras.
        """
        # Neural Net
        self.input_boards = Input(shape=(self.len_history, self.board_x, self.board_y))
        x_image = Reshape((self.len_history, self.board_x, self.board_y))(self.input_boards)

        np.random.seed(0)
        h_conv1 = Activation('relu')(BatchNormalization()(Conv2D(self.numFilters, self.convsize, padding='same',data_format='channels_first')(x_image)))
        h_conv2 = Activation('relu')(BatchNormalization()(Conv2D(self.numFilters, self.convsize, padding='same',data_format='channels_first')(h_conv1)))
        h_conv3 = Activation('relu')(BatchNormalization()(Conv2D(self.numFilters, 3, padding='valid',data_format='channels_first')(h_conv2)))
        h_conv4 = Activation('relu')(BatchNormalization()(Conv2D(self.numFilters, 3, padding='valid',data_format='channels_first')(h_conv3)))

        h_conv4_flat = Flatten()(h_conv4)
        s_fc1 = Dropout(self.dropout)(Activation('sigmoid')(BatchNormalization()(Dense(64)(h_conv4_flat))))
        s_fc2 = Dropout(self.dropout)(Activation('sigmoid')(BatchNormalization()(Dense(64)(s_fc1))))
        self.pi = Dense(self.action_size, activation='softmax', name='pi')(s_fc2)
        self.v = Dense(1, activation='tanh')(Dense(1, activation='linear', name='v')(s_fc2))

        # Assemble the model
        self.model = Model(inputs=self.input_boards, outputs=[self.pi, self.v])
        self.model.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=Adam(self.lr, clipnorm=1))

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

    def predict(self, board_history):
        """
        Predict the policy vector and value for a given board.

        Inputs
        ------
        board: np array representing the game board
        """
        board_history = board_history[np.newaxis, :, :, :]
        pi, v = self.model.predict(board_history)
        return pi[0], v[0][0]

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
