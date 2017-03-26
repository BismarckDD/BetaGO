from keras.models import Sequential
from keras.layers import convolutional
from keras.layers.core import Dense, Flatten
import keras.backend as K

### Parameters obtained from paper ###
# K = 152                        # depth of convolutional layers
# LEARNING_RATE = .003           # initial learning rate
# DECAY = 8.664339379294006e-08  # rate of exponential learning_rate decay


class CNNValue(object):

    """ Uses a Convolutional Neural Network to evaluate the state of a game
        and compute a probability distribution over the next action
    """

    @staticmethod
    def create_network(**kwargs):

        model = Sequential()

        model.add(convolutional.Convolution2D(
            input_shape=(49, 19, 19), nb_filter=K, nb_row=5, nb_col=5,
            init='uniform', activation='relu', border_mode='same'))

        for i in range(2, 13):
            model.add(convolutional.Convolution2D(
                nb_filter=K, nb_row=3, nb_col=3,
                init='uniform', activation='relu', border_mode='same'))

        model.add(convolutional.Convolution2D(
            nb_filter=1, nb_row=1, nb_col=1,
            init='uniform', activation='linear', border_mode='same'))

        model.add(Flatten())
        model.add(Dense(256, init='uniform'))
        model.add(Dense(1, init='uniform', activation="tanh"))

        return model


if __name__ == '__main__':
    print "Wait to implement"
# TODO command line instantiation.
