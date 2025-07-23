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


def compute_rewards(states, observations, infos):
    # states: [batch, obs_dim], observations: [batch, obs_dim], infos: list of dicts
    # Ball, Player1, Player2 위치 인덱스: 0:2, 4:6, 8:10
    d1 = torch.norm(states[:, 0:2] - states[:, 4:6], dim=1)
    d2 = torch.norm(states[:, 0:2] - states[:, 8:10], dim=1)
    d3 = torch.norm(observations[:, 0:2] - observations[:, 4:6], dim=1)
    d4 = torch.norm(observations[:, 0:2] - observations[:, 8:10], dim=1)

    reward1 = torch.tensor([int(info["winner"] == "Player1") for info in infos], device=config.device) * 10
    reward2 = torch.tensor([int(info["winner"] == "Player2") for info in infos], device=config.device) * 10

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
    # 진행 중인 환경만 transition 저장
    for i in range(len(states)):
        if not done[i]:
            agent1.store_transition(states[i].unsqueeze(0), actions1[i].unsqueeze(0), next_states[i].unsqueeze(0), rewards[i,0].view(1,1))
            agent2.store_transition(states[i].unsqueeze(0), actions2[i].unsqueeze(0), next_states[i].unsqueeze(0), rewards[i,1].view(1,1))


def simulate_step(env, agent1, agent2, states, done, render=True):
    # states: [batch, obs_dim]
    performance_log.log_performance("simulate_step_select_action")
    actions1 = agent1.select_action(states)
    actions2 = agent2.select_action(states)
    actions = torch.cat([actions1, actions2], dim=1)
    
    performance_log.log_performance("simulate_step_env_step")
    observations, reward, terminated, truncated, infos = env.step(actions)
    
    performance_log.log_performance("simulate_step_render")
    if render:
        frames = env.render()
        render_env(frames)
        
    performance_log.log_performance("simulate_step_compute_rewards")
    rewards = compute_rewards(states, observations, infos)
    step_done = (terminated | truncated)
    # 이미 done인 환경은 state를 그대로 유지, 아닌 경우에만 next_states를 업데이트
    next_states = states.clone()
    mask = ~done & ~step_done
    next_states[mask] = observations[mask]
    return next_states, rewards, step_done, actions1, actions2


def optimize_agents(agent1, agent2):
    agent1.optimize_model()
    agent2.optimize_model()
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
        done = torch.zeros(batch_size, dtype=torch.bool, device=config.device)
        for t in count():
            print(f"{t}\r", end="")
            performance_log.log_performance("train_loop_step")
            next_states, rewards, step_done, actions1, actions2 = simulate_step(env, agent1, agent2, states, done, render)
            
            performance_log.log_performance("train_loop_store_transitions")
            store_transitions(agent1, agent2, states, actions1, actions2, next_states, rewards, done)
            
            performance_log.log_performance("train_loop_optimize_model")
            agent1.optimize_model()
            agent2.optimize_model()
            
            performance_log.log_performance("train_loop_soft_update_target")
            agent1.soft_update_target()
            agent2.soft_update_target()
            
            performance_log.log_performance("train_loop_update_done_mask")
            # update done mask
            done = done | step_done
            for i in range(batch_size):
                if done[i] and not step_done[i]:
                    plot.episode_durations.append(t + 1)
                    plot.plot_durations()
                    print(f"env {i} 보상: Player1={rewards[i,0].item():.2f}, Player2={rewards[i,1].item():.2f}")
            states = next_states
            if done.all():
                break
        
        performance_log.print_performance()
    print('Complete')
    plot.plot_durations(show_result=True)
    cv2.destroyAllWindows()


def run_simulation_loop(env, agent1, agent2, render=True):
    agent1.set_eval_mode(True)
    agent2.set_eval_mode(True)
    
    state, info = env.reset()
    state = state.unsqueeze(0) if state.ndim == 1 else state
    
    for t in count():
        next_state, rewards, done = simulate_step(env, agent1, agent2, state, torch.zeros(env.num_envs, dtype=torch.bool, device=config.device), render)
        state = next_state
        if done:
            print(f"Episode finished in {t+1} steps. Rewards: {rewards}")
            break 