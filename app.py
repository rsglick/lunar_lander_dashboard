import os
from multiprocessing import Pool

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import streamlit as st

import gym
from stable_baselines3 import PPO, SAC, TD3


# This is required since Gym is dumb and forces a new window for render.
os.system("Xvfb :1 -screen 0 1024x768x24 &")
os.environ['DISPLAY'] = ':1'


titleString = "Continous Gym Environment Dashboard"
st.set_page_config(
    page_title=titleString,
    # page_icon=":)",
    layout="wide",
    initial_sidebar_state="collapsed" #  ("auto" or "expanded" or "collapsed")
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
env_name = st.selectbox('Select Environment', env_list, index=2)
training_timesteps = st.number_input("Training Timesteps: ", value=10, min_value=10, max_value=10000, step=1000)
video_frame_length = st.number_input("Testing  Timesteps: ", value=10, min_value=10, max_value=2000, step=100)

def eval_agent(model=None):
    env = gym.make(env_name)
    vid = []

    obs = env.reset()
    for i in range(video_frame_length):
        if model is not None:
            action, _states = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        vid.append(env.render(mode="rgb_array"))
        if done:
            obs = env.reset()
    env.close()
    return vid
    
def create_vid(vid, vid_name):
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

    anim.save(f'{vid_name}.mp4', fps=60)
    
def worker(col, rl_alg_name, rl_alg):
    col.header(f"{rl_alg_name} Agent")

    if rl_alg is not None:
        with st.spinner(f'Training ({rl_alg_name}) Agent in Process...'):
            env = gym.make(env_name)
            model = rl_alg("MlpPolicy", env, verbose=1)
            model.learn(total_timesteps=training_timesteps)
    else:
        model = None

    with st.spinner(f'Testing ({rl_alg_name}) Agent in Process...'):
        vid = eval_agent(model=model)
        create_vid(vid, rl_alg_name)
        with open(f'{rl_alg_name}.mp4', 'rb') as vid:
            vid_bytes = vid.read()
        col.video(vid_bytes)

rl_alg_dict = {
    "PPO": PPO,
    "SAC": SAC,
    "TD3": TD3,
    "Random": None,
}
# rl_alg_name = st.selectbox('RL Algorithm: PPO / SAC / TD3:',rl_alg_dict.keys())

cols = st.columns(4)

for col, (rl_alg_name, rl_alg) in zip(cols, rl_alg_dict.items()):
    worker(col, rl_alg_name, rl_alg)



# starmap_iter = []
# for col, (rl_alg_name, rl_alg) in zip(cols, rl_alg_dict.items()):
#     starmap_iter.append([col, rl_alg_name, rl_alg])
# # st.write(starmap_iter)
# with Pool() as pool:
#     pool.starmap(worker, starmap_iter)
    



# vid = eval_agent(model=None)
# create_vid(vid, "random")
# with open('random.mp4', 'rb') as vid:
#     vid_bytes = vid.read()
# col[-1].video(vid_bytes)






st.sidebar.markdown("---")
st.sidebar.markdown(
"""
[<img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' class='img-fluid' width=50 height=50>]
(https://github.com/rsglick/lunar_lander_dashboard) <small> Dashboard Beta </small>""",
unsafe_allow_html=True,
)