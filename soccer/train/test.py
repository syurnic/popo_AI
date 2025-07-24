import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))  # soccer/train 디렉토리 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import model
import agent
import config
import episode_loops

import soccer.soccer_env  # 환경 등록
from parallel_env import ParallelVecSoccerEnv

import performance_log
performance_log.clear_performance()

NUM_ENVS = 1

def main():
    print(config.device)

    env_fns = [lambda: FlattenObservation(gym.make("SoccerEnv", render_mode="rgb_array")) for _ in range(NUM_ENVS)]
    vec_env = ParallelVecSoccerEnv(env_fns)
    agent1 = agent.DQNAgent(model.DQN, vec_env.envs[0].observation_space, vec_env.envs[0].action_space[0], label="Player1")
    agent2 = agent.DQNAgent(model.DQN, vec_env.envs[0].observation_space, vec_env.envs[0].action_space[1], label="Player2")

    agent1.load_model(f"./soccer/train/models/v{config.MODEL_VERSION}-1.pth")
    agent2.load_model(f"./soccer/train/models/v{config.MODEL_VERSION}-2.pth")

    episode_loops.run_simulation_loop(vec_env, agent1, agent2, render=True)

if __name__ == "__main__":
    main()