import os

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import streamlit as st

import gym

os.system("Xvfb :1 -screen 0 1024x768x24 &")
os.environ['DISPLAY'] = ':1'

titleString = "Lunar Lander Dashboard"
st.set_page_config(
    page_title=titleString,
    # page_icon=":)",
    # layout="wide",
    initial_sidebar_state="expanded"
)


st.title(titleString)

## https://github.com/openai/gym/blob/master/gym/envs/__init__.py
env_list =[
    # "CartPole-v1",
    "Pendulum-v0",
    "MountainCarContinuous-v0",
    "LunarLanderContinuous-v2",
    "BipedalWalkerHardcore-v3",
]
env_name = env_list[2]

# from stable_baselines3 import PPO, SAC
# env = gym.make(env_name)
# model = PPO("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=10)

env = gym.make(env_name)
vid = []

obs = env.reset()
for i in range(100):
    # action, _states = model.predict(obs, deterministic=True)
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    # env.render()
    vid.append(env.render(mode="rgb_array"))
    if done:
      obs = env.reset()
env.close()



fig = plt.figure()
im = plt.imshow(vid[0], interpolation='none', aspect='auto', vmin=0, vmax=1)

def animate(i):
    im.set_array(vid[i])
    return [im]
anim = animation.FuncAnimation(
                               fig, 
                               animate, 
                               frames = len(vid),
                               interval = len(vid) / 60, # in ms
                               )

anim.save('test_anim.mp4', fps=60)

with open('test_anim.mp4', 'rb') as vid:
    vid_bytes = vid.read()
st.video(vid_bytes)







st.sidebar.markdown("---")
st.sidebar.markdown(
"""
[<img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' class='img-fluid' width=50 height=50>]
(https://github.com/rsglick/lunar_lander_dashboard) <small> Dashboard Beta </small>""",
unsafe_allow_html=True,
)