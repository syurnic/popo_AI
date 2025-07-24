import gymnasium as gym
import torch
import numpy as np
import cv2
from itertools import count
import config
import agent
import model
import plot
from soccer.soccer_env import SoccerEnv
import performance_log
import time
import matplotlib.pyplot as plt
import random

# 파일 상단에 loss_history 리스트 추가
loss_history = []

def compute_rewards(states, observations, infos):
    # states: [batch, obs_dim], observations: [batch, obs_dim], infos: list of dicts
    # Ball, Player1, Player2 위치 인덱스: 0:2, 4:6, 8:10
    d1 = torch.norm(states[:, 0:2] - states[:, 4:6], dim=1)
    d2 = torch.norm(states[:, 0:2] - states[:, 8:10], dim=1)
    d3 = torch.norm(observations[:, 0:2] - observations[:, 4:6], dim=1)
    d4 = torch.norm(observations[:, 0:2] - observations[:, 8:10], dim=1)

    reward1 = torch.tensor([int(info["winner"] == "Player1") for info in infos], device=states.device) * 10
    reward2 = torch.tensor([int(info["winner"] == "Player2") for info in infos], device=states.device) * 10

    reward1, reward2 = reward1 - reward2, reward2 - reward1
    reward1 = reward1 + (d1 - d3)
    reward2 = reward2 + (d2 - d4)
    rewards = torch.stack([reward1, reward2], dim=1)
    return rewards


def render_env(frames):
    # frames: 각 환경의 render() 결과 리스트
    window_w, window_h = 500, 300
    positions = [(0, 0), (520, 0), (0, 340), (520, 340)]  # 2x2 격자 배치 예시
    for i, frame in enumerate(frames):
        if frame is not None:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame_bgr = cv2.resize(frame_bgr, (window_w, window_h))
            win_name = f'Env_{i}'
            cv2.imshow(win_name, frame_bgr)
            if i < len(positions):
                x, y = positions[i]
                cv2.moveWindow(win_name, x, y)
    cv2.waitKey(1)


def store_transitions(agent1, agent2, states, actions1, actions2, next_states, rewards, done):
    # 진행 중인 환경만 transition 저장 (배치)
    mask = ~done
    if mask.any():
        agent1.store_transition(states[mask], actions1[mask], next_states[mask], rewards[mask, 0].unsqueeze(1))
        agent2.store_transition(states[mask], actions2[mask], next_states[mask], rewards[mask, 1].unsqueeze(1))


def simulate_step(env, agent1, agent2, states, done, render=True):
    # states: [batch, obs_dim]
    performance_log.log_performance("simulate_step_select_action")
    actions1 = agent1.select_action(states, use_cpu=True)
    actions2 = agent2.select_action(states, use_cpu=True)
    actions = torch.cat([actions1, actions2], dim=1)
    
    performance_log.log_performance("simulate_step_env_step")
    observations, rewards, terminated, truncated, infos = env.step(actions)
    
    performance_log.log_performance("simulate_step_render")
    if render:
        frames = env.render()
        render_env(frames)
    
    performance_log.log_performance("simulate_step_update_target_net_cpu")
    if random.random() < 0.1:
        agent1.update_target_net_cpu()
        agent2.update_target_net_cpu()

    performance_log.log_performance("simulate_step_compute_rewards")
    rewards = compute_rewards(states, observations, infos)
    step_done = (terminated | truncated)
    # 이미 done인 환경은 state를 그대로 유지, 아닌 경우에만 next_states를 업데이트
    next_states = states.clone()
    mask = ~done & ~step_done
    next_states[mask] = observations[mask]
    return next_states, rewards, step_done, actions1, actions2

def optimize_agents_loop(agent1, agent2, stop_event):
    global loss_history
    while not stop_event.is_set():
        loss1 = agent1.optimize_model()
        loss2 = agent2.optimize_model()
        if loss1 is not None and loss2 is not None:
            loss_history.append((loss1 + loss2) / 2)
            
        agent1.soft_update_target()
        agent2.soft_update_target()

def train_loop(env, agent1, agent2, num_episodes=100, render=True, reset=True):
    if reset:
        agent1.reset_steps()
        agent2.reset_steps()
    
    agent1.set_eval_mode(False)
    agent2.set_eval_mode(False)
    
    batch_size = env.num_envs
    
    for i_episode in range(num_episodes):

        states, infos = env.reset()
        done = torch.zeros(batch_size, dtype=torch.bool, device=states.device)
        step_counters = torch.zeros(batch_size, dtype=torch.long)
        episode_steps = []

        for t in count():
            print(f"{t}\r", end="")

            performance_log.log_performance("train_loop_step")
            next_states, rewards, step_done, actions1, actions2 = simulate_step(
                env, agent1, agent2, states, done, render
            )

            performance_log.log_performance("train_loop_store_transitions")
            store_transitions(
                agent1, agent2, states, actions1, actions2, next_states, rewards, done
            )

            performance_log.log_performance("train_loop_update_done_mask")
            step_counters += (~done).long()
            newly_done = (~done) & step_done

            if newly_done.any():
                for idx in torch.where(newly_done)[0]:
                    episode_steps.append(step_counters[idx].item())

            done = done | step_done
            states = next_states

            if done.all():
                avg_steps = float(np.mean(episode_steps)) if episode_steps else 0.0
                plot.episode_durations.append(avg_steps)
                plot.plot_durations(loss_history=loss_history)
                break

            if t % 2 == 0:
                plt.pause(0.001)
            
            time.sleep(0.001)

        performance_log.print_performance()
        config.MODEL_VERSION += 1
        agent1.save_model(f"./soccer/train/models/v{config.MODEL_VERSION}-1.pth")
        agent2.save_model(f"./soccer/train/models/v{config.MODEL_VERSION}-2.pth")
    print('Complete')
    plot.plot_durations(show_result=True, loss_history=loss_history)
    cv2.destroyAllWindows()

def run_simulation_loop(env, agent1, agent2, render=True):
    agent1.set_eval_mode(True)
    agent2.set_eval_mode(True)
    
    batch_size = env.num_envs
    
    states, info = env.reset()
    done = torch.zeros(batch_size, dtype=torch.bool, device=states.device)
    
    for t in count():
        next_states, rewards, step_done, actions1, actions2 = simulate_step(env, agent1, agent2, states, done, render)
        states = next_states
        if step_done.all():
            print(f"Episode finished in {t+1} steps. Rewards: {rewards}")
            break 