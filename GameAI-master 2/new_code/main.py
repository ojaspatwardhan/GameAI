import gym
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from collections import Counter, deque
from statistics import mean,median
import agent

LR = 2e-3

# env = gym.make('CartPole-v0')
# env.reset()
# goal_steps = 500
# score_requirement = 50
# initial_games = 10000

# mario env
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = BinarySpaceToDiscreteSpaceEnv(env, COMPLEX_MOVEMENT)
env.reset()
goal_steps = 5000
score_requirement = 100
initial_games = 1000
game_memory = deque(maxlen=2000)
gamma = 0.95    # discount rate
epsilon = 0.7  # exploration rate
epsilon_min = 0.01
epsilon_decay = 0.995
learning_rate = 0.001





def random_games():
    for episode in range(5):
        print("the game we are playing is: ",episode)
        env.reset()
        for t in range(goal_steps):
            env.render()
            action = env.action_space.sample()
            observation, rewards, done, info = env.step(action)
            if done:
                break






def initial_population():
    training_data = []
    scores = []
    accepted_scores = []

# game
    for _ in range(initial_games):
        print("the game we are playing is: ", _)
        score = 0
        game_memory = []
        prev_observation = []

        for _ in range(goal_steps):
            #env.render()
            action = env.action_space.sample()
            # print("the action taking place is: ",action)
            observation, rewards, done, info = env.step(action)

            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action])


            prev_observation = observation

            score += rewards

            if done:
                break


        #print("the game memory is: ", game_memory)

        if score >= score_requirement:
            accepted_scores.append(score)


            # print("the values in game memory is", game_memory[0])
            # print ("hehehehhe: ", game_memory[0][1])

            print("getting trained for this iteration")


            for data in game_memory:
                # print("checking whats in data 1", data[1])
                if data[1] == 0:
                    output = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                elif data[1] == 1:
                    output = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                elif data[1] == 2:
                    output = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                elif data[1] == 3:
                    output = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
                elif data[1] == 4:
                    output = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
                elif data[1] == 5:
                    output = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
                elif data[1] == 6:
                    output = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
                elif data[1] == 7:
                    output = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
                elif data[1] == 8:
                    output = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
                elif data[1] == 9:
                    output = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
                elif data[1] == 10:
                    output = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
                elif data[1] == 11:
                    output = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]



                #print("the data received is: ", data[0])
                training_data.append([data[0], output])

        env.reset()
        scores.append(score)

    training_data_save = np.array(training_data)
    np.save('saved.npy', training_data_save)


    # print('Avergae accepted score:', mean(accepted_scores))
    # print('Median accepted score:', median(accepted_scores))
    print (Counter(accepted_scores))

    return training_data



def neural_network_model(input_size):
    print ("inside neural network\n")
    network = input_data(shape = [None, input_size,3], name='input')

    network = fully_connected(network, 64, activation='relu')

    network = fully_connected(network, 128, activation='relu')

    network = fully_connected(network, 128, activation='relu')

    network = fully_connected(network, 64, activation='relu')
    network = dropout(network, 0.4)

    network = fully_connected(network, 12, activation='softmax')

    network = regression(network, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(network, tensorboard_dir='log')

    return model


def train_model(training_data, model = False):
    print("inside train model\n")


    print(training_data)

    X = np.array([i[0][0] for i in training_data] ).reshape(-1,len(training_data[0][0][0]),3)
    Y = [i[1] for i in training_data]

    if not model:
        model = neural_network_model(input_size= len(X[0]))

    print("back to nn\n")
    model.fit({'input': X}, {'targets': Y}, n_epoch=3, snapshot_step=500, show_metric=True, run_id='openaistuff')

    return model


def load_model():

    model = neural_network_model(256)

    model.load("training_model.model")


    return model





def remember(self, state, action, reward, next_state, done):
    self.memory.append((state, action, reward, next_state, done))


def replay(batch_size):
    global epsilon
    minibatch = random.sample(game_memory, batch_size)
    for state, action, reward, next_state, done in minibatch:
        target = reward
        if not done:
            target = reward + gamma * np.amax(model.predict(next_state)[0])
            print("Target:", target, "Reward:", reward)
        else:
            print("Here")
            target = reward * -100
            # exit(0)
        target_f = model.predict(state)
        target_f[0][action] = target
        model.fit(state, target_f, n_epoch=1)
        # model.save("model.h5")
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay




training_data = initial_population()
model = train_model(training_data)
model.save("training_model.model")



#model = train_model()





def act(state):
    if np.random.rand() <= epsilon:
        x = random.randrange(7)
        return x
    act_values = model.predict(state)
    # print(act_values)
    return np.argmax(act_values[0])




scores = []
choices = []

for each_game in range(50):
    score = 0
    # game_memory = []
    prev_obs = []
    env.reset()
    for _ in range(goal_steps):
        env.render()
        # env.step(0)
        if len(prev_obs) == 0:
            action = random.randrange(7)
        else:
            #print (prev_obs," :>>> ",len(prev_obs[0]))
            #exit(0)
            # action = np.argmax(model.predict(prev_obs.reshape(-1,len(prev_obs[0]),3))[0])
            state = prev_obs.reshape(-1,len(prev_obs[0]),3)
            action = act(state)
            # print(model.predict(prev_obs.reshape(-1,len(prev_obs[0]),3))[0])
            # print(action)

            # if action == 6:
            #     action = 1

        choices.append(action)

        # print("the action is: ", action)
        new_observation, reward, done, info = env.step(action)
        prev_obs = new_observation
        # game_memory.append([new_observation, action])
        game_memory.append((prev_obs, action, reward, new_observation, done))
        score += reward
        if done:
            break
    replay(32)
    scores.append(score)

    print('Average Score:>>> ',sum(scores)/len(scores))
    print ('Choices 1: {}, Choices 0: {}'.format(choices.count(1)/len(choices), choices.count(0)/len(choices)))























