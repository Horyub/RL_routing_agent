import gymnasium as gym
import routing_env
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import random
import torch
from torch import nn
import yaml

from experience_replay import ReplayMemory
from dqn import DQN

from datetime import datetime, timedelta
import argparse
import itertools

import os


# Directory for saving run info
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)
# 'Agg': used to generate plots as images and save them to a file instead of rendering to screen
matplotlib.use('Agg')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu' # force cpu

class Agent():

    def __init__(self, hyperparameter_set, adjacency_matrix :np.array):
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]
            

        self.hyperparameter_set = hyperparameter_set
        self.adjacency_matrix = adjacency_matrix

        self.env_id             = hyperparameters['env_id']
        self.learning_rate_a    = hyperparameters['learning_rate_a']        # learning rate (alpha)
        self.discount_factor_g  = hyperparameters['discount_factor_g']      # discount rate (gamma)
        self.network_sync_rate  = hyperparameters['network_sync_rate']      # number of steps the agent takes before syncing the policy and target network
        self.replay_memory_size = hyperparameters['replay_memory_size']     # size of replay memory
        self.mini_batch_size    = hyperparameters['mini_batch_size']        # size of the training data set sampled from the replay memory
        self.epsilon_init       = hyperparameters['epsilon_init']           # 1 = 100% random actions
        self.epsilon_decay      = hyperparameters['epsilon_decay']          # epsilon decay rate
        self.epsilon_min        = hyperparameters['epsilon_min']            # minimum epsilon value
        self.stop_on_reward     = hyperparameters['stop_on_reward']         # stop training after reaching this number of rewards
        self.fc1_nodes          = hyperparameters['fc1_nodes']
      
        self.loss_fn = nn.MSELoss()          
        self.optimizer = None                

        # Path to Run info
        self.LOG_FILE   = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.png')
        
        
    def run(self, is_training=True, render=False):
        if is_training:
            start_time = datetime.now()
            last_graph_update_time = start_time

            log_message = "Training starting..."
            print(log_message)
            with open(self.LOG_FILE, 'w') as file:
                file.write(log_message + '\n')
        
            env = gym.make(self.env_id, render_mode='human' if render else None, adjacency_matrix=self.adjacency_matrix)
        else: 
            n = self.adjacency_matrix.shape[0]
            total_requests = n * (n-1)
            env = gym.make(self.env_id, render_mode='human' if render else None, adjacency_matrix=self.adjacency_matrix, num_traffic_demands=total_requests, ramp_up=False)

        env = gym.wrappers.FlattenObservation(env)

        num_actions = env.action_space.n

        num_states = env.observation_space.shape[0] 
   
        rewards_per_episode = []

        # Create policy and target network. Number of nodes in the hidden layer can be adjusted.
        policy_dqn = DQN(num_states, num_actions, self.fc1_nodes).to(device)

        if is_training:
            
            epsilon = self.epsilon_init
            memory = ReplayMemory(self.replay_memory_size)

            # Create the target network and make it identical to the policy network
            target_dqn = DQN(num_states, num_actions, self.fc1_nodes).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())

            # Policy network optimizer
            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

            
            epsilon_history = []
            # Track number of steps taken. Used for syncing policy => target network.
            step_count=0

        else:
            # Load learned policy
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))

            # switch model to evaluation mode
            policy_dqn.eval()

        # Train INDEFINITELY, manually stop the run when you are satisfied (or unsatisfied) with the results
        episode_count = 0
        for episode in itertools.count():

            state, info = env.reset()  
            state = torch.tensor(state, dtype=torch.float, device=device) 
            terminated = False     
            truncated = False      
            episode_reward = 0.0    


            while(not terminated and not truncated and episode_reward < self.stop_on_reward):
                # Select action based on epsilon-greedy
                if is_training and random.random() < epsilon:
                    # Select a random valid action.
                    action_value = random.choice(info["valid_actions"])
                    action = torch.tensor(
                        action_value,
                        dtype=torch.int64,
                        device=device,
                    )
                else:
                    with torch.no_grad():
                        q_values = policy_dqn(
                            state.unsqueeze(dim=0)
                        ).squeeze()

                        action_mask = torch.tensor(
                            info["action_mask"],
                            dtype=torch.bool,
                            device=device,
                        )

                        q_values[~action_mask] = float("-inf")
                        action = q_values.argmax()

                # Execute action and keep the next state's action mask for learning.
                new_state,reward,terminated,truncated,info = env.step(action.item())

                episode_reward += reward

                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)

                if is_training:
                    done = terminated or truncated
                    next_action_mask = torch.tensor(
                        info["action_mask"],
                        dtype=torch.bool,
                        device=device,
                    )

                    # Save experience into memory
                    memory.append((
                        state,
                        action,
                        new_state,
                        reward,
                        done,
                        next_action_mask,
                    ))

                    step_count+=1

                state = new_state

            rewards_per_episode.append(episode_reward)

            if is_training:
                
                if episode % 100 == 0:
                    mean_reward = np.mean(rewards_per_episode[-100:])

                    log_message = (
                        f"Episode {episode} | "
                        f"Reward: {episode_reward:.2f} | "
                        f"Mean reward: {mean_reward:.2f} | "
                        f"Epsilon: {epsilon:.4f}"
                    )

                    print(log_message)

                    with open(self.LOG_FILE, "a") as file:
                        file.write(log_message + "\n")
                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
        
                    

                # Update graph every 10 seconds.
                current_time = datetime.now()

                if current_time - last_graph_update_time > timedelta(seconds=10):
                    self.save_graph(rewards_per_episode, epsilon_history)
                    last_graph_update_time = current_time

                # If enough experience has been collected.
                if len(memory) > self.mini_batch_size:
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_dqn, target_dqn)

                    epsilon = max(
                        epsilon * self.epsilon_decay,
                        self.epsilon_min,
                    )
                    epsilon_history.append(epsilon)

                    if step_count > self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        step_count = 0


    def save_graph(self, rewards_per_episode, epsilon_history):
        # Save plots
        fig = plt.figure(1)
        # Plot average rewards (Y-axis) vs episodes (X-axis)
        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x-99):(x+1)])
        plt.subplot(121) # plot on a 1 row x 2 col grid, at cell 1
        # plt.xlabel('Episodes')
        plt.ylabel('Mean Rewards')
        plt.plot(mean_rewards)
        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        plt.subplot(122) # plot on a 1 row x 2 col grid, at cell 2
        # plt.xlabel('Time Steps')
        plt.ylabel('Epsilon Decay')
        plt.plot(epsilon_history)

        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        # Save plots
        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)



    def optimize(self, mini_batch, policy_dqn, target_dqn):
    
        states, actions, new_states, rewards, dones, next_action_masks = zip(*mini_batch)

        # Stack tensors to create batch tensors
        states = torch.stack(states)

        actions = torch.stack(actions)

        new_states = torch.stack(new_states)

        rewards = torch.stack(rewards)
        dones = torch.tensor(dones).float().to(device)
        next_action_masks = torch.stack(next_action_masks)

        with torch.no_grad():
            # Calculate target Q values using only valid next-state actions.
            next_q_values = target_dqn(new_states)
            next_q_values[~next_action_masks] = float("-inf")

            max_next_q_values = next_q_values.max(dim=1)[0]
            no_valid_next_actions = ~next_action_masks.any(dim=1)
            max_next_q_values[no_valid_next_actions] = 0.0

            target_q = (
                rewards
                + (1 - dones)
                * self.discount_factor_g
                * max_next_q_values
            )

        # Calcuate Q values from current policy
        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()
      

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()  
        loss.backward()            
        self.optimizer.step()    

        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('hyperparameters', help='')
    parser.add_argument('--train', help='Training mode', action='store_true')
    args = parser.parse_args()

    import numpy as np

    adjacency_matrix = np.array([
        [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
        [1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
        [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        ], dtype=np.float64)


    dql = Agent(hyperparameter_set=args.hyperparameters, adjacency_matrix=adjacency_matrix)

    if args.train:
        dql.run(is_training=True)
    else:
        dql.run(is_training=False, render=True)
