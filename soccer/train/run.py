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

def main():
    env = gym.make("SoccerEnv", render_mode="rgb_array")
    env = FlattenObservation(env)
    state, info = env.reset()
    agent1 = agent.DQNAgent(model.DQN, env.observation_space, env.action_space[0], label="Player1")
    agent2 = agent.DQNAgent(model.DQN, env.observation_space, env.action_space[1], label="Player2")
    
    episode_loops.train_loop(env, agent1, agent2, num_episodes=1, render=True)
    #episode_loops.run_simulation_loop(env, agent1, agent2, render=False)

if __name__ == "__main__":
    main()