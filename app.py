import os
import base64
import pathlib

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import streamlit as st

import gym
from stable_baselines3 import PPO, SAC, TD3

# CONSTANTS
DEFAULT_FLOAT = -9999999.9
RL_ALG_DICT = {
"PPO": PPO,
"SAC": SAC,
"TD3": TD3,
"Random": None,
}

## https://github.com/openai/gym/blob/master/gym/envs/__init__.py
ENV_LIST =[
# "CartPole-v1",
"Pendulum-v0",
"MountainCarContinuous-v0",
"LunarLanderContinuous-v2",
"BipedalWalkerHardcore-v3",
]

###

def sidebar_footer():
    st.sidebar.markdown("---")
    st.sidebar.markdown(
    """
    [<img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' class='img-fluid' width=50 height=50>]
    (https://github.com/rsglick/lunar_lander_dashboard) <small> Dashboard Beta </small>""",
    unsafe_allow_html=True,
    )

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

class Agent:
    def __init__(self, env_name=None, rl_alg_name="Random"):
        self.env_name     = env_name
        self.rl_alg_name  = rl_alg_name
        
        self.video_rgb_array = [] 
        self.gif_url = None

    @property
    def rl_alg(self):
        return RL_ALG_DICT[self.rl_alg_name]

    @property
    def np_results_filepath(self):
        return pathlib.Path(f"{self.rl_alg_name}_{self.env_name}_eval_results.npz")

    @property
    def gif_path(self):
        return pathlib.Path(f"{self.rl_alg_name}_{self.env_name}.gif")

    @property
    def already_ran_bool(self):
        return self.gif_path.exists() and self.np_results_filepath.exists()

    @property
    def total_reward(self):
        if self.already_ran_bool:
            total_reward = np.load(self.np_results_filepath)["total_reward"]
        else:
            total_reward = np.array([DEFAULT_FLOAT])
        return total_reward

    @property
    def reward_threshold(self):
        if self.already_ran_bool:
            reward_threshold = np.load(self.np_results_filepath)["reward_threshold"]
        else:
            reward_threshold = np.array([DEFAULT_FLOAT])
        return reward_threshold

    @property
    def model(self):
        if self.rl_alg_name == "Random":
           return "Random"

        model_path = pathlib.Path(f"./models/model{rl_alg_name}_{env_name}.zip")
        if model_path.exists():
            return rl_alg.load(model_path.as_posix())
        else:
            return None

    @property
    def model_valid(self):
        return self.rl_alg_name == "Random" or self.model is not None


    def test_agent(self):
        if self.already_ran_bool:
            return

        env = gym.make(self.env_name)
        reward_threshold = env.spec.reward_threshold
        if reward_threshold is None:
            reward_threshold = np.nan

        vid = []
        total_reward = []

        done = False
        episode_reward = 0  
        obs = env.reset()

        while done is False:
            if self.model == "Random":
                action = env.action_space.sample()
            else:
                action, _states = self.model.predict(obs, deterministic=True)
                
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            vid.append(env.render(mode="rgb_array"))
            if done:
                total_reward.append(episode_reward)
                obs = env.reset()
        env.close()

        self.video_rgb_array = vid
        # self.total_reward = np.array(total_reward)
        # self.reward_threshold = reward_threshold        
        np.savez(self.np_results_filepath, total_reward=total_reward, reward_threshold=reward_threshold)

    def create_gif(self):
        if self.already_ran_bool:
            return

        fig = plt.figure()
        im = plt.imshow(self.video_rgb_array[0], interpolation='none', aspect='auto', vmin=0, vmax=1)

        def animate(i):
            im.set_array(self.video_rgb_array[i])
            return [im]
        
        fps = 30
        anim = animation.FuncAnimation(
                                    fig, 
                                    animate, 
                                    frames = len(self.video_rgb_array),
                                    interval = len(self.video_rgb_array) / fps, # in ms
                                    )

        writer = animation.ImageMagickWriter(fps=fps)
        anim.save(self.gif_path, writer=writer)

    def write_gif_file(self):     
        file_ = open(self.gif_path, "rb")
        contents = file_.read()
        gif_url = base64.b64encode(contents).decode("utf-8")
        file_.close()
        self.gif_url = gif_url
 
st.title(titleString)

# env_name = st.selectbox('Select Environment', ENV_LIST, index=2)
env_name = ENV_LIST[2]
st.header(f"{env_name} - https://gym.openai.com/envs/{env_name}/")

rl_algs_chosen = st.multiselect('Select RL Agents:', RL_ALG_DICT.keys() )
rl_alg_used_dict = {key:value for key, value in RL_ALG_DICT.items() if key in rl_algs_chosen}
run_agents_bool = st.checkbox('Run Agents', value=False)

if run_agents_bool and rl_alg_used_dict:
    cols = st.columns(len(rl_alg_used_dict))

    for col, (rl_alg_name, rl_alg) in zip(cols, rl_alg_used_dict.items()):
        col.header(f"{rl_alg_name} Agent")

        agent = Agent(env_name, rl_alg_name)

        with st.spinner(f'Testing ({agent.rl_alg_name}) Agent ...'):
            agent.test_agent()

        with st.spinner(f'Creating ({agent.rl_alg_name}) Agent GIF  ...'):
            agent.create_gif()
            agent.write_gif_file()    

        col.write(f"Reward Threshold: {agent.reward_threshold:.3f}")
        col.write(f"Reward: {agent.total_reward.mean():.3f} +/- {agent.total_reward.std():.3f}")
        col.markdown(
            f'<img src="data:image/gif;base64,{agent.gif_url}" alt="RL GIF" width=400 height=400>',
            unsafe_allow_html=True,
        ) 

sidebar_footer()