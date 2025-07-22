import gymnasium as gym
import torch
import numpy as np
import cv2
from itertools import count
import config
import agent
import model
import plot


def compute_rewards(state, observation, info):
    d1 = torch.norm(state[0][0:2] - state[0][4:6]).item()
    d2 = torch.norm(state[0][0:2] - state[0][8:10]).item()
    d3 = np.linalg.norm(observation[0:2] - observation[4:6])
    d4 = np.linalg.norm(observation[0:2] - observation[8:10])
    reward1 = int(info["winner"] == "Player1") * 10
    reward2 = int(info["winner"] == "Player2") * 10
    reward1, reward2 = (reward1 - reward2, reward2 - reward1)
    reward1 += (d1 - d3)
    reward2 += (d2 - d4)
    return torch.tensor([reward1, reward2], device=config.device)


def render_env(env):
    frame = env.render()
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow('Env', frame_bgr)
    cv2.waitKey(1)


def store_transitions(agent1, agent2, state, action1, action2, next_state, reward):
    agent1.store_transition(state, action1, next_state, reward[0].view(1, 1))
    agent2.store_transition(state, action2, next_state, reward[1].view(1, 1))


def simulate_step(env, agent1, agent2, state, render=True):
    action1 = agent1.select_action(state)
    action2 = agent2.select_action(state)
    
    observation, reward, terminated, truncated, info = env.step((action1.item(), action2.item()))
    
    if render:
        render_env(env)
    
    rewards = compute_rewards(state, observation, info)
    
    done = terminated or truncated
    if terminated:
        next_state = None
    else:
        next_state = torch.tensor(observation, dtype=torch.float32, device=config.device).unsqueeze(0)
    
    store_transitions(agent1, agent2, state, action1, action2, next_state, rewards)
    
    return next_state, rewards, done

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
    
    for i_episode in range(num_episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=config.device).unsqueeze(0)
        
        for t in count():
            next_state, rewards, done = simulate_step(env, agent1, agent2, state, render)
            
            optimize_agents(agent1, agent2)
            
            state = next_state
            
            if done:
                plot.episode_durations.append(t + 1)
                plot.plot_durations()
                print(f"보상: Player1={rewards[0].item():.2f}, Player2={rewards[1].item():.2f}")
                break
    
    print('Complete')
    plot.plot_durations(show_result=True)
    cv2.destroyAllWindows()

def run_simulation_loop(env, agent1, agent2, render=True):
    agent1.set_eval_mode(True)
    agent2.set_eval_mode(True)
    
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=config.device).unsqueeze(0)
    
    for t in count():
        next_state, rewards, done = simulate_step(env, agent1, agent2, state, render)
        state = next_state
        if done:
            print(f"Episode finished in {t+1} steps. Rewards: {rewards}")
            break 