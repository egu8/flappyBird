import logging
import sys

import gym
from gym.wrappers import Monitor
import gym_ple

from Agents import DQN


def transformGameState(state):
    import numpy as np

    player_y = state['player_y']
    player_vel = state['player_vel']
    next_pipe_dist_to_player = state['next_pipe_dist_to_player']
    next_pipe_top_y = state['next_pipe_top_y']
    next_pipe_bottom_y = state['next_pipe_bottom_y']

    next_next_pipe_dist_to_player = state['next_next_pipe_dist_to_player']
    next_next_pipe_top_y = state['next_next_pipe_top_y']
    next_next_pipe_bottom_y = state['next_next_pipe_bottom_y']

    delta_top = player_y - next_pipe_top_y
    delta_bottom = player_y - next_pipe_bottom_y

    return np.array([delta_bottom, delta_top, player_vel, next_pipe_dist_to_player])

if __name__ == '__main__':

    env = gym.make('FlappyBird-v0' if len(sys.argv) < 2 else sys.argv[1])

    outdir = './tmp'
    env = Monitor(env, directory=outdir, force=True)

    env.seed(0)
    dqn = DQN(env)

    import torch
    dqn.eval_net.load_state_dict(torch.load("200_weights.pt",map_location=torch.device('cpu')))
    dqn.target_net.load_state_dict(torch.load("200_weights.pt",map_location=torch.device('cpu')))

    episode_count = 1001
    reward = 0
    done = False
    too_low = False

    for i in range(episode_count):
        print('\nCollecting experience...')

        s = env.reset()
        s = transformGameState(env.reset())

        import pygame
        clock = pygame.time.Clock()

        ep_r = 0
        while True:

            a = dqn.choose_action(s)
            if too_low:
                a = 0
            # take action
            s_, r, done, info = env.step(a)
            too_low = (s_['player_y'] > 330)
            s_ = transformGameState(s_)

            dqn.store_transition(s, a, r, s_)

            ep_r += r
            if dqn.memory_counter > dqn.memory_capacity:
                loss = dqn.learn()
                if done:
                    print('Ep: ', i,
                          '| Ep_r: ', round(ep_r, 2))

            if done:
                break
            s = s_

            clock.tick(30)
            env.render()

        if i % 200 == 0:
            import torch
            torch.save(dqn.eval_net.state_dict(), str(i) + "_weights.pt")

        print("episode: " + str(i) + " DONE")

    # Dump result info to disk
    env.close()