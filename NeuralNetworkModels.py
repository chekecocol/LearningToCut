
### Neural Network models developed by
### Ezequiel Piedras, 2022

import tensorflow as tf
import numpy as np


class Policy(object):

    def __init__(self, state_dim, lr, hidden_size):

        # Using Keras Functional API to build my custom model
        input1 = tf.keras.Input(shape=(state_dim,1))
        input2 = tf.keras.Input(shape=(state_dim,1))

        lstm1 = tf.keras.layers.LSTM(8)(input1)
        lstm2 = tf.keras.layers.LSTM(8)(input2)
        dense1a = tf.keras.layers.Dense(hidden_size, activation = "tanh")(lstm1)
        dense2a = tf.keras.layers.Dense(hidden_size)(dense1a)
        dense1b = tf.keras.layers.Dense(hidden_size, activation = "tanh")(lstm2)
        dense2b = tf.keras.layers.Dense(hidden_size)(dense1b)

        output = tf.reduce_mean(tf.tensordot(dense2a,tf.transpose(dense2b), axes = 1), axis = 0)

        self.model = tf.keras.Model(inputs = [input1,input2], outputs = output)

        # DEFINE THE OPTIMIZER
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = lr)

        # RECORD HYPER-PARAMS
        self.state_dim = state_dim

    def call(self, state):

        # Unwrap state tuple
        A = state[0]
        b = state[1]
        # Normalize constraints
        A = np.divide(A.transpose(), b).transpose()
        A = np.divide(A,(np.max(A,axis=0)-np.min(A,axis=0))) - np.min(A,axis=0)
        A = tf.expand_dims(A, axis = 2)
        A = tf.cast(A, tf.double)

        D = state[3]
        e = state[4]
        # Normalize cuts
        D = np.divide(D.transpose(), e).transpose()
        D = np.divide(D, (np.max(D, axis=0) - np.min(D, axis=0))) - np.min(D, axis=0)
        D = tf.expand_dims(D, axis = 2)
        D = tf.cast(D, tf.double)

        # Call model
        scores = self.model([A,D])
        return scores


    def compute_prob(self, state):

        prob = tf.cast(tf.nn.softmax(self.call(state), axis = -1), tf.double)  # probabilities computed as softmax, sum to 1
        return prob.numpy()

    def train(self, states, actions, Es):

        with tf.GradientTape() as tape:
            # COMPUTE probability vector pi(s) for all s in states
            total_loss = 0
            N = len(states)

            for i in range(N):

                prob = tf.cast(tf.nn.softmax(self.call(states[i]), axis=-1), tf.double)
                action_onehot = tf.cast(tf.one_hot(actions[i], len(states[i][-1])), tf.double)
                prob_selected = tf.reduce_sum(prob * action_onehot, axis=-1)

                # FOR ROBUSTNESS
                prob_selected += 1e-8

                total_loss += -1 * Es[i] * tf.math.log(prob_selected)

            # BACKWARD PASS
            total_loss_2 = total_loss / N
            gradients = tape.gradient(total_loss_2, self.model.trainable_variables)

            # UPDATE
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return total_loss_2.numpy()


# Helper function
def discounted_rewards(r, gamma):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_sum = 0
    for i in reversed(range(0, len(r))):
        discounted_r[i] = running_sum * gamma + r[i]
        running_sum = discounted_r[i]
    return list(discounted_r)