import torch
import os
import numpy as np
import sys
sys.path.insert(0, '')

from env.path_task.path_task_v0 import raw_env
from agent import PPOAgent
from pprint import pprint

TRAJECTORY_LEN = 256 # how many transitions each agent's trajectory should hold
PPO_EPOCHS = 3 # how many times we train per set of trajectories
MAX_UPDATES = 3000
GAMMA = 0.99
CLIP_EPS = 0.2
LR = 3e-4
ENTROPY_COEFF = 0.01

def train(env, path):
    observations, _ = env.reset()
    agents = env.possible_agents

    observation_space = env.observation_space()
    act_dim = env.action_space().n
    
    model = PPOAgent(observation_space, act_dim, LR, GAMMA, CLIP_EPS, ENTROPY_COEFF) 

    for update in range(MAX_UPDATES):
        print("At update " + str(update))

        obs_buf = {agent: {"grid": [], "one_hot": []} for agent in agents}
        act_buf = {agent: [] for agent in agents}
        logp_buf = {agent: [] for agent in agents}
        rew_buf = {agent: [] for agent in agents}
        done_buf = {agent: [] for agent in agents}
        mask_buf = {agent: [] for agent in agents}
        val_buf = []
        all_obs_buf = []

        for _ in range(TRAJECTORY_LEN):

            obs_tensor = {
                "grid": torch.tensor(np.array([observations[a]["observation"]["grid"] for a in agents])),
                "one_hot": torch.tensor(np.array([observations[a]["observation"]["one_hot"] for a in agents])),
            }

            masks_tensor = torch.tensor(
                np.array([observations[a]["action_mask"] for a in agents]),
                dtype=torch.float32,
            )

            all_obs = torch.tensor(observations["agent_0"]["observation"]["grid"]) # each agent sees global state, so just get from first agent
            all_obs_buf.append(all_obs)

            actions, logprobs, all_states_value = model.act(obs_tensor, all_obs.unsqueeze(0), masks_tensor)

            action_dict = {
                agent: actions[i].item()
                for i, agent in enumerate(agents)
            }

            next_obs, rewards, terminations, truncations, infos = env.step(action_dict)

            for i, agent in enumerate(agents):
                obs_buf[agent]["grid"].append(obs_tensor["grid"][i])
                obs_buf[agent]["one_hot"].append(obs_tensor["one_hot"][i])
                act_buf[agent].append(actions[i])
                logp_buf[agent].append(logprobs[i])
                rew_buf[agent].append(rewards[agent])
                mask_buf[agent].append(masks_tensor[i])
                done_buf[agent].append(terminations[agent] or truncations[agent])
            val_buf.append(all_states_value)

            observations = next_obs

            if not env.agents:
                observations, _ = env.reset()

        rtgs_buf = {}
        advantages_buf = {}

        # convert to tensor to feed into network
        val_buf = torch.stack(val_buf)
        all_obs_buf = torch.stack(all_obs_buf)
        
        for agent in agents:
            obs_buf[agent]["grid"] = torch.stack(obs_buf[agent]["grid"])
            obs_buf[agent]["one_hot"] = torch.stack(obs_buf[agent]["one_hot"]) 
            mask_buf[agent] = torch.stack(mask_buf[agent])
            act_buf[agent] = torch.stack(act_buf[agent])
            logp_buf[agent] = torch.stack(logp_buf[agent])
            rew_buf[agent] = torch.tensor(rew_buf[agent], dtype=torch.float32)
            done_buf[agent] = torch.tensor(done_buf[agent], dtype=torch.float32)
            rtgs_buf[agent] = model.compute_rtgs(rew_buf[agent], done_buf[agent])
            advantages_buf[agent] = model.compute_advantages(rtgs_buf[agent], val_buf)

        for _ in range(PPO_EPOCHS):
            aggr_critic_loss = 0.0
            aggr_actor_loss = 0.0
            
            all_values = model.critic(all_obs_buf).squeeze(-1) # with centralized critic, only need to calculate this once for all agents
            
            # compute loss for each individual agent, and sum up
            for i, agent in enumerate(agents):
                critic_loss, actor_loss = model.compute_loss(all_values, obs_buf[agent], act_buf[agent], logp_buf[agent], 
                                                             advantages_buf[agent], rtgs_buf[agent], mask_buf[agent])
                aggr_critic_loss += critic_loss
                aggr_actor_loss += actor_loss

            # get mean loss
            aggr_critic_loss /= len(agents)
            aggr_actor_loss /= len(agents)
            model.optimize(aggr_critic_loss, aggr_actor_loss)

    env.close()

    model.save(path)

def evaluate(env, model_path, episodes=5):
    observations, _ = env.reset()
    agents = env.possible_agents

    observation_space = env.observation_space()
    act_dim = env.action_space().n
    
    model = PPOAgent(observation_space, act_dim) 
    model.load_actor(model_path)
    model.set_actor_eval()

    episode_returns = []

    for _ in range(episodes):
        observations, _ = env.reset()
        agents = env.agents[:]

        episode_reward = {agent: 0.0 for agent in agents}

        while env.agents:
            obs_tensor = {
                "grid": torch.tensor(np.array([observations[a]["observation"]["grid"] for a in agents])),
                "one_hot": torch.tensor(np.array([observations[a]["observation"]["one_hot"] for a in agents])),
            }

            masks_tensor = torch.tensor(
                np.array([observations[a]["action_mask"] for a in agents]),
                dtype=torch.float32,
            )

            with torch.no_grad():
                logits = model.actor(obs_tensor)
                logits = logits.masked_fill(masks_tensor == 0, -torch.inf)
                actions = torch.argmax(logits, dim=-1)
            actions_dict = {
                agent: actions[i].item()
                for i, agent in enumerate(agents)
            }

            observations, rewards, terminations, truncations, infos = env.step(actions_dict)
            
            for agent in agents:
                episode_reward[agent] += rewards[agent]

        episode_returns.append(np.mean(list(episode_reward.values())))

    mean_return = np.mean(episode_returns)
    std_return = np.std(episode_returns)
    print(f"Mean return: {mean_return:.2f} ± {std_return:.2f}")


if __name__ == "__main__":
    save_path: str = os.path.dirname(os.path.abspath(__file__)) + "/models/path_task_v0.pt"
    #train(raw_env(render_mode=None, delay_btw_frames=0, with_task_infos=False), save_path)
    #evaluate(raw_env(render_mode="human", delay_btw_frames=0.5, with_task_infos=True), save_path, 1)