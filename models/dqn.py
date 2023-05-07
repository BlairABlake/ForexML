import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import numpy as np
import random
from collections import deque

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 

# DQN Agent
class DQNAgent:
    def __init__(self, state_size=4, action_size=3, duration=10):
        self.state_size = state_size
        self.action_size = action_size
        self.duration = duration
        self.memory = deque([], maxlen=200)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.model.summary()

    def _build_model(self):
        model = tf.keras.Sequential()
        model.add(Input(shape=(self.duration, self.state_size)))
        model.add(Bidirectional(LSTM(16)))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss="mae", optimizer=Adam(learning_rate=self.learning_rate))
        return model


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = np.expand_dims(state, axis=0)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        
        # Extract states and next_states from the minibatch
        states = np.array([x[0] for x in minibatch])
        next_states = np.array([x[3] for x in minibatch])

        # Predict the Q-values for current states and next states
        current_q_values = self.model.predict(states)
        next_q_values = self.model.predict(next_states)

        # Update the Q-values based on the rewards and next Q-values
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            target = reward
            if not done:
                target += self.gamma * np.amax(next_q_values[i])
            current_q_values[i][action] = target

        # Fit the model to the updated Q-values
        self.model.fit(states, current_q_values, batch_size=batch_size, epochs=1, verbose=2)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)