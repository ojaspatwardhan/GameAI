from agent import DQNAgent
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import feature_extraction
import numpy as np

if __name__ == "__main__":
    # initialize gym environment and the agent
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = BinarySpaceToDiscreteSpaceEnv(env, COMPLEX_MOVEMENT)
    env.reset()
    agent = DQNAgent()
    max_x_position = -1
    time = 0
    # Iterate the game
    episodes = 1
    for e in range(episodes):
        print(e)
        # reset state in the beginning of each game
        state = env.reset()
        state = feature_extraction.get_image_content(state)
        state = np.expand_dims(state, axis=0)
        # time_t is each frame of the game
        # the more time_t the more score
        for time_t in range(100):
            # print("Here")
            # turn this on if you want to render
            env.render()
            # Decide action
            action = agent.act(state)
            print("Action", action)
            # if action > 12:
            #     continue
            # Advance the game to the next frame based on the action.
            next_state, reward, done, _ = env.step(action)
            # if _['x_pos'] > max_x_position:
            #     max_x_position = _['x_pos']
            #     time = _['time']
            # elif time - _['time'] >= 10:
            #     done=True
            next_state = feature_extraction.get_image_content(next_state)
            next_state = np.expand_dims(next_state, axis=0)
            # Adding reward
            # reward = reward * 2 if not done else -10
            # Remember the previous state, action, reward, and done
            agent.remember(state, action, reward, next_state, done)
            # make next_state the new current state for the next frame.
            state = next_state
            # done becomes True when the game ends
            if done:
                # print the score and break out of the loop
                print("episode: {}/{}, score: {}, x_pos: {}"
                      .format(e, episodes, time_t, _['x_pos']))
                break
        # train the agent with the experience of the episode
        print("Episode done")
        agent.replay(32)
