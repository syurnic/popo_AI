import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))  # soccer/train 디렉토리 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import gymnasium as gym
from gymnasium.wrappers import FlattenObservation

import threading
import time

import model
import agent
import config
import episode_loops

import soccer.soccer_env  # 환경 등록
from parallel_env import ParallelVecSoccerEnv

import performance_log
performance_log.clear_performance()

performance_log.USE_LOG = True

NUM_ENVS = 64

def main():
    print(config.device)

    env_fns = [lambda: FlattenObservation(gym.make("SoccerEnv", render_mode="rgb_array")) for _ in range(NUM_ENVS)]
    vec_env = ParallelVecSoccerEnv(env_fns)
    agent1 = agent.DQNAgent(model.DQN, vec_env.envs[0].observation_space, vec_env.envs[0].action_space[0], label="Player1")
    agent2 = agent.DQNAgent(model.DQN, vec_env.envs[0].observation_space, vec_env.envs[0].action_space[1], label="Player2")

    # 모델 버전에 맞는 파일이 있으면 불러오기
    model_path1 = f"./soccer/train/models/v{config.MODEL_VERSION}-1.pth"
    model_path2 = f"./soccer/train/models/v{config.MODEL_VERSION}-2.pth"
    if os.path.exists(model_path1):
        print(f"모델 로드: {model_path1}")
        agent1.load_model(model_path1)
    if os.path.exists(model_path2):
        print(f"모델 로드: {model_path2}")
        agent2.load_model(model_path2)

    stop_event = threading.Event()
    optimizer_thread = threading.Thread(target=episode_loops.optimize_agents_loop, args=(agent1, agent2, stop_event))
    optimizer_thread.start()

    try:
        episode_loops.train_loop(vec_env, agent1, agent2, num_episodes=100, render=False)
    except KeyboardInterrupt:
        print("\n[강제종료 감지: 모델 저장 및 스레드 종료]")

        config.MODEL_VERSION += 1
        print(f"모델 버전: {config.MODEL_VERSION}")

        agent1.save_model(f"./soccer/train/models/v{config.MODEL_VERSION}-1.pth")
        agent2.save_model(f"./soccer/train/models/v{config.MODEL_VERSION}-2.pth")

        import matplotlib.pyplot as plt
        plt.close('all')
    finally:
        stop_event.set()
        optimizer_thread.join()

if __name__ == "__main__":
    main()