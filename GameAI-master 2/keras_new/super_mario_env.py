from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env.reset()
env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)

# done = True
# for step in range(5000):
#     if done:
#         state = env.reset()
#     state, reward, done, info = env.step(env.action_space.sample())
#     env.render()
#
# env.close()

# for i_episode in range(20):
#     state = env.reset()
#     for t in range(100):
#         state, reward, done, info = env.step(env.action_space.sample())
#         env.render()
#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             break
# env.close()

for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
