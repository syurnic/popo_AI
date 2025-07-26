import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))  # soccer/train 디렉토리 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import soccer.soccer_env  # 환경 등록
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import model
import agent
import config
import cv2
import keyboard
import torch

def main():
    print(config.device)

    env_fns = [lambda: FlattenObservation(gym.make("SoccerEnv", render_mode="rgb_array"))]
    vec_env = env_fns[0]()
    obs, info = vec_env.reset()

    # AI 에이전트 준비 (Player2)
    ai_agent = agent.DQNAgent(
        model.DQN,
        vec_env.observation_space,
        vec_env.action_space[0],  # Player2의 action_space
        label="Player1"
    )
    ai_agent.load_model(f"./soccer/train/models/v{config.MODEL_VERSION}-1.pth")

    ai_agent.set_eval_mode(True)

    print("Player: a(왼), d(오), w(위), space(슛)")
    print("종료: q")

    for i in range(1000):
        if keyboard.is_pressed("q"):
            break

        frame = vec_env.render()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('Env', frame_bgr)
        cv2.waitKey(1)

        # Player: 키보드 입력
        action1 = 0
        if keyboard.is_pressed("a"):
            action1 = 1
        elif keyboard.is_pressed("d"):
            action1 = 2
        elif keyboard.is_pressed("w"):
            action1 = 3
        elif keyboard.is_pressed("space"):
            action1 = 4

        # Player2: AI
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        action2 = ai_agent.select_action(obs_tensor, use_cpu=True).item()

        action = [action2, action1]

        obs, reward, terminated, truncated, info = vec_env.step(action)

        print(obs[2])

        if terminated or truncated:
            print(info["winner"])
            break

if __name__ == "__main__":
    main()
    cv2.waitKey(100)