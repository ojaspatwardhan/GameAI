from keras.layers import Dense,Flatten
from keras.models import Sequential
import random
import numpy as np
from keras.optimizers import Adam
from collections import deque
from keras import backend as K
from keras.applications import VGG16,MobileNet

# Deep Q-learning Agent
class DQNAgent:
    def __init__(self):
        self.count = 0
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model

        model = Sequential()
        model.add(Flatten())
        model.add(Dense(24, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(12, activation="linear"))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        try:
            model.load_weights("model.h5")
        except:
            pass
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            x = random.randrange(12)
            return x
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
              target = reward + self.gamma * np.amax(self.model.predict(next_state))
            target_f = self.model.predict(state)
            print(target_f.shape)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
            self.model.save("model.h5")
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay