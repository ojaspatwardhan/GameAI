from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import numpy as np
import json
import time

import gym


env = gym.make('SuperMarioBros-v0')

for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        observation = env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()






# np.set_printoptions(threshold=np.inf)

#
#
# def take_input():
# #6 ---> left
# #5 maybe jump
# #4 lower jump across
# #3 run
# #2 higher jump across
# # #1 move right normal
#     return input()
#
#
# def render_game(env):
#
#     env.render()
#
#
# def deallocate_resource(env):
#     env.close()
#
#
# def game_loop(state, env):
#     while True:
#         state, reward, done, info = env.step(0)
#         render_game(env)
#         movement = take_input()
#         for i in range(20):
#             print("inside game loop the movement is ", movement)
#             state, reward, done, info = env.step(movement)
#             print(state)
#             # print(reward)
#             # with open('state.json', 'w') as file:
#             #     json.dump(state.tolist(), file)
#             time.sleep(0.01)
#             render_game(env)
#
#
#
#
# def game():
#
#     env = gym_super_mario_bros.make('SuperMarioBros-v0')
#     env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)
#
#     for step in range(15000):
#         state = env.reset()
#
#         game_loop(state, env)
#
#
#
#     deallocate_resource(env);
#
#
# if __name__ == '__main__':
#     game()