import torch
import numpy as np
import pandas as pd
import math
from enviroment import  Environment
from DiffusionActor import DiffusionActor
from Agent import TD3
from memory import ReplayBuffer
from model import DoubleCritic
from numpy import random

env = Environment()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = 1.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

agent = TD3(
    state_dim=state_dim,
    action_dim=action_dim,
    max_action=max_action,
    actor_class=DiffusionActor,
    critic_class=DoubleCritic,
    device=device
)

replay_buffer = ReplayBuffer(state_dim, action_dim)

episodes = 10000
max_steps = 150
batch_size = 512
policy_noise = 0.05
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = episodes

log = {
    "Episode": [],
    "AvgReward": [],
    "Actor Loss": [],
    "Critic Loss": [],
    "Energy": [],
    "AoI": []
}

for episode in range(episodes):
    state = env.reset()
    ep_reward, ep_energy, ep_aoi = 0, 0, 0
    ep_actor, ep_critic = 0, 0

    epsilon = epsilon_end+(epsilon_start-epsilon_end)*math.exp(-1.0 * episode/30)

    for step in range(max_steps):
        if np.random.rand() < epsilon:
            action = np.zeros(action_dim)
            for n in range(action_dim):
                action[n] = random.uniform(-1, 1)

            #print("Action Noisy",action)
        else:
            action = agent.select_action(np.array(state))
            noise = np.random.normal(0, policy_noise, size=action_dim)
            action = np.clip(action + noise, -max_action, max_action)
            #print("action policy",action)

        next_state, reward, done, energy, aoi = env.step(action)
        #print("rewards",step,reward)
        replay_buffer.add(state, action, reward, next_state, float(done))
        state = next_state

        ep_reward += reward
        ep_energy += energy
        ep_aoi += aoi

        if len(replay_buffer) >= batch_size:
            info = agent.train(replay_buffer, batch_size)
            ep_critic += info["critic_loss"]
            ep_actor += 0 if info["actor_loss"] is None else info["actor_loss"]

        if done:
            break

    log["Episode"].append(episode)
    log["AvgReward"].append(ep_reward / max_steps)
    log["Actor Loss"].append(ep_actor / max_steps)
    log["Critic Loss"].append(ep_critic / max_steps)
    log["Energy"].append(ep_energy / max_steps)
    log["AoI"].append(ep_aoi)

    print(f"Ep {episode+1:3d} | Reward: {ep_reward/max_steps:.2f} | ε = {epsilon:.2f}")
print(log)
df = pd.DataFrame(log)
df.to_csv("training_diffusion_transformer_log.csv", index=False)
print("Training complete. Logs saved.")
