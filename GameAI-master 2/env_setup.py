from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
import numpy as np
from agent import DQNAgent
from keras.models import model_from_json
import feature_extraction


env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = BinarySpaceToDiscreteSpaceEnv(env, COMPLEX_MOVEMENT)
agent = DQNAgent()


def load_model():
    # load json and create model

    json_file = open('model.json', 'r')

    loaded_model_json = json_file.read()

    json_file.close()

    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model

    loaded_model.load_weights("model_best.h5")

    print("Loaded model from disk")

    loaded_model.compile(loss='mse', optimizer='sgd')

    return loaded_model

model = load_model()

episodes = 100
for e in range(episodes):
    print(e)
    # reset state in the beginning of each game
    state = env.reset()
    state = feature_extraction.get_image_content(state)
    state = np.expand_dims(state, axis=0)
    # time_t is each frame of the game
    # the more time_t the more score
    for time_t in range(50000):

        env.render()

        action = agent.act(state)

        next_state, reward, done, _ = env.step(action)
        next_state = feature_extraction.get_image_content(next_state)
        next_state = np.expand_dims(next_state, axis=0)

        state = next_state

        if done:
            # print the score and break out of the loop
            print("episode: {}/{}, score: {}, x_pos: {}"
                  .format(e, episodes, time_t, _['x_pos']))
            break
    # train the agent with the experience of the episode
    print("Episode done, count is: ", agent.count)


# for i_episode in range(20):
#     state = env.reset()
#     state = feature_extraction.get_image_content(state)
#     state = np.expand_dims(state, axis=0)
#     for t in range(5000):
#         actions = model.predict(state)
#         observation, reward, done, info = env.step(int(np.argmax(actions[0])))
#         env.render()
#         state = observation
#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             break
# env.close()






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