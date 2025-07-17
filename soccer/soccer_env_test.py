import soccer_env
import gymnasium as gym
from matplotlib import pyplot as plt
import cv2
import keyboard

env = gym.make("SoccerEnv", render_mode="rgb_array")

env.reset()

for i in range(1000):
    if keyboard.is_pressed("q"):
        break
    
    frame = env.render()
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow('Env', frame_bgr)
    cv2.waitKey(1000 // int(env.metadata.get('render_fps')))
    
    action = [0, 0]
    if keyboard.is_pressed("a"):
        action[0] = 1
    elif keyboard.is_pressed("d"):
        action[0] = 2
    elif keyboard.is_pressed("w"):
        action[0] = 3
    elif keyboard.is_pressed("space"):
        action[0] = 4
    
    if keyboard.is_pressed("left"):
        action[1] = 1
    elif keyboard.is_pressed("right"):
        action[1] = 2
    elif keyboard.is_pressed("up"):
        action[1] = 3
    elif keyboard.is_pressed("enter"):
        action[1] = 4
    
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        print(info["winner"])
        break