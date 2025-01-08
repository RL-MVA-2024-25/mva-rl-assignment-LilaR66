from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient

#env = TimeLimit(
#    env=HIVPatient(domain_randomization=False), max_episode_steps=200)
# The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.

# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!

#class ProjectAgent:
#    def act(self, observation, use_random=False):
#        return 0
#
#    def save(self, path):
#        pass
#
#    def load(self):
#        pass

#**************************************************
# ***************** MY CODE BELOW *****************
#**************************************************

# Implementation of a simple DQN Network 
#**************************************************
import os
import random
from copy import deepcopy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.wrappers import TimeLimit

from env_hiv import HIVPatient
from evaluate import evaluate_HIV, evaluate_HIV_population

#**************************************************

class simpleDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        Deep Q-Network avec des couches fully connected.

        Args:
            input_dim (int): Dimension de l'espace d'observation (6 pour HIVPatient).
            output_dim (int): Nombre d'actions possibles (4 pour HIVPatient).
        """
        super(simpleDQN, self).__init__()
        hidden_size = 256

        # Couche d'entrée vers la première couche cachée
        self.fc1 = nn.Linear(input_dim,   hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.fc6 = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        """
        Passe avant du réseau.

        Args:
            x (torch.Tensor): Tensor contenant l'état de l'environnement.

        Returns:
            torch.Tensor: Q-values pour chaque action.
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = self.fc6(x) # Output layer (Q-values)
        return x

#**************************************************

class ReplayBuffer:
    """
    A Replay Buffer to store and sample transitions for DQN training.
    The buffer stores [state, action, reward, next_state, done] list and
    supports sampling mini-batches for training.

    This implementation includes conversion of sampled data into PyTorch tensors.
    """
    def __init__(self, capacity, device):
        """
        Initialize the Replay Buffer.

        Args:
            capacity (int): The maximum number of transitions the buffer can store.
                           When the capacity is reached, older transitions are replaced
                           using a First-In-First-Out (FIFO) approach.
        """
        self.capacity = capacity  # Maximum size of the buffer
        self.data = [] # List to store transitions
        self.index = 0 # Circular index for the next position to insert data
        self.device = device

    def append(self, s, a, r, s_, d):
        """
        Add a new transition to the buffer.

        Args:
            s (np.ndarray): Current state of the environment.
            a (int): Action taken by the agent.
            r (float): Reward received after taking the action.
            s_ (np.ndarray): Next state after taking the action.
            d (bool): Done flag indicating if the episode has ended.
        """
        if len(self.data) < self.capacity:
            self.data.append(None) # Add a placeholder if buffer is not full
        self.data[self.index] = (s, a, r, s_, d)  # Store the new transition
        self.index = (self.index + 1) % self.capacity  # Move to the next position (circular buffer)

    def sample(self, batch_size):
        """
        Sample a random batch of transitions from the buffer.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            tuple: A tuple containing batches of states, actions, rewards,
                   next_states, and done flags, all as PyTorch tensors.
        """
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))

    def __len__(self):
        """
        Get the current size of the buffer.

        Returns:
            int: The number of transitions currently stored in the buffer.
        """
        return len(self.data)

#**************************************************  
     
class ProjectAgent:
    def __init__(self):
        """
        Initialize the DQN agent with parameters from the config file.
        """
        config = CONFIG

        # State normalization values (hardcoded)
        self.state_mean = np.array([3.64168114e+05, 7.83013522e+03, 2.80073042e+02, 3.32431142e+01,
                                     3.71611071e+04, 5.60337695e+01], dtype=np.float32)
        self.state_std = np.array([1.28716475e+05, 1.43828518e+04, 3.42618001e+02, 2.49548152e+01,
                                    6.98382448e+04, 3.58961009e+01], dtype=np.float32)
        # Reward normalization values (hardcoded)
        self.reward_mean = 46535.77882020019
        self.reward_std = 36955.77656083679


        # Device configuration: GPU if available, else CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Configuration
        self.input_dim = config['input_dim']    # Dimension of state space
        self.output_dim = config['output_dim']  # Dimension of action space
        self.gamma = config['gamma']            # Discount factor
        self.lr = config['learning_rate']
        self.nb_gradient_steps = config ['nb_gradient_steps']
        self.target_update_steps = config['target_update_steps']
        # Replay buffer
        self.memory = ReplayBuffer(config['buffer_size'], self.device)
        self.batch_size = config['batch_size']
        # epsilon greedy strategy
        self.epsilon_max = config['epsilon_max']
        self.epsilon = self.epsilon_max                    # Current exploration rate.
        self.epsilon_min = config['epsilon_min']           # Minimum value of epsilon.
        self.epsilon_step = config['epsilon_step']         # Decrement for epsilon at each step (for linear decay).
        self.epsilon_delay = config['epsilon_delay_decay'] # Number of steps before starting epsilon decay.
        self.save_path = config['save_path'] # path to save the best model
        self.full_state_path = config['full_state_path'] # path to save the full model state

        # Initialize networks
        #    Initialize the main Q-network (used for predicting Q-values during training and action selection).
        self.model = simpleDQN(self.input_dim, self.output_dim).to(self.device)
        #    Initialize the target Q-network. This network will be used to compute the stable target Q-values.
        #    Copy the weights and biases from the main Q-network to the target network.
        #    This ensures both networks start with the same parameters initially.
        self.target_model = deepcopy(self.model).to(self.device)
        #    Set the target network to evaluation mode (no gradients will be computed).
        #    This prevents unnecessary gradient calculations, as the target network is not trained directly.
        self.target_model.eval()

        # Criterion
        self.criterion = torch.nn.SmoothL1Loss() #torch.nn.MSELoss() # Provides the expectation sum of rewards
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # Step counter for target network update
        self.step_count = 0
        # Initialisation
        self.previous_val = 0
        self.episode_returns = []


    def act(self, state):
        """
        Choose the best action based on the model's predictions (pure exploitation).
        Method act is called during interaction with the environment to select
        n action based on the current observation (without updating network
        weights).

        Args:
            state (array): The current state of the environment.

        Returns:
            int: The index of the best action.
        """
        # Use the model to predict Q-values for the current state
        # Disable gradient calculation since we don't need backpropagation during action selection
        with torch.no_grad():
            # Convert the current state into a PyTorch tensor for model input
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            # unsqueeze(0) adds a batch dimension, as the model expects input in batch format
            q_values = self.model(state_tensor)  # Predict Q-values for the state
        return torch.argmax(q_values).item()  # Return the action with the highest Q-value (exploitation)


    def select_action(self, state, env, strategy="epsilon_greedy"):
        """
        Decide between exploration and exploitation based on a given strategy.
        - strategy="epsilon_greedy" : uses an epsilon-greedy strategy with linear epsilon decay.
        - strategy="random" : choose a random action

        Args:
            state (array): The current state of the environment.
            env (gym.Env): The environment, used to sample random actions.
            strategy (str): The exploration/exploitation strategy to use. Default is "epsilon_greedy".

        Returns:
            int: The index of the chosen action.
        """
        if strategy == "epsilon_greedy":
          # Delay epsilon decay until training_steps > epsilon_delay
          if self.step_count > self.epsilon_delay:
              self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_step)
              ##print("self.epsilon_min", self.epsilon_min)
              #print("self.epsilon - self.epsilon_step", self.epsilon - self.epsilon_step)
          if np.random.rand() < self.epsilon:
              # Exploration: choose a random action using the environment's action space
              return env.action_space.sample()
          else:
              # Exploitation: choose the best action using the model
              return self.act(state)
        elif strategy == "random":
            # Completely random policy
            return env.action_space.sample()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")


    def gradient_step(self):
        """
        Perform a single gradient step to update the Q-network.
        """
        if len(self.memory) > self.batch_size:
            # Sample a batch of transitions (state, action, reward, next_state, done) from the replay buffer
            states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

            # Compute Q-values for the actions chosen in the sampled transitions
            # The model predicts Q-values for all actions, gather(1, actions) extracts Q-values for the selected actions
            q_values = self.model(states).gather(1, actions.to(torch.long).unsqueeze(1))

            # Compute the target Q-values using the target network
            # Max Q-value for the next state
            max_next_q_values = self.target_model(next_states).max(1)[0].detach() # detach() = disconnect from the computational graph.
            # Bellman equation for the target Q-values:
            # target Q = reward + gamma * max(Q(next_state, a')) if not done
            target_q_values = torch.addcmul(rewards, 1-dones, max_next_q_values, value=self.gamma)

            # Compute the loss between predicted Q-values and target Q-values
            loss = self.criterion(q_values, target_q_values.unsqueeze(1))

            # Perform backpropagation and update the main network's weights
            self.optimizer.zero_grad()  # Reset gradients to zero
            loss.backward()             # Compute gradients through backpropagation
            self.optimizer.step()       # Update weights using the optimizer (Adam)


    def train(self, env, num_episodes):
        """
        Train the agent in the given environment.

        This method is responsible for :
        - Sampling Replay Buffer transitions.
        - Calculating target Q-values with the target network.
        - Updating main network weights to minimize the error between
        predicted Q-values and targets.

        Args:
            env (gym.Env): The environment to train the agent in.
            num_episodes (int) : Number of episodes to train the agent.

        Returns:
            episode_returns (list) : total rewards per episode
        """
        # Pass model to device (especially pass model to GPU if the model is loaded from CPU)
        self.target_model.to(self.device)
        self.model.to(self.device)

        print("\nStart Training...")
        # ------------- BEGIN loop 1
        for episode in range(num_episodes):
            state, _ = env.reset()
            done = False
            truncated = False
            episode_rewards = 0

            # ------------- BEGIN loop 2
            while not done and not truncated:
                # Epsilon-greedy action selection
                action = self.select_action(state, env, strategy="epsilon_greedy")

                # Step in the environment
                next_state, reward, done, truncated, _ = env.step(action)

                # Normalize before adding to buffer
                #state = self.normalize_state(state)
                #next_state = self.normalize_state(next_state)
                #reward = (reward - self.reward_mean) / self.reward_std

                # Store transition in replay buffer
                self.memory.append(state, action, reward, next_state, done)

                # Update the model
                for _ in range(self.nb_gradient_steps):
                    self.gradient_step()

                # Update target network periodically
                if self.step_count % self.target_update_steps == 0:
                    # Copy weights from the main network to the target network
                    self.target_model.load_state_dict(self.model.state_dict())
                    #print(f"Target network updated at step {self.step_count}")

                # Update state,reward and counters
                self.step_count +=1 # Increment the step counter
                state = next_state
                episode_rewards += reward
            # ------------- END loop 2

            # Compute validation score at the end of the episode
            validation_score = evaluate_HIV(agent=self, nb_episode=1)

            # Save best model
            if validation_score > self.previous_val:
                    self.previous_val = validation_score
                    self.best_model = deepcopy(self.model).to(self.device)
                    self.save(self.save_path)
                    self.save_full_state(self.full_state_path)

            # print to log progress
            print(f"Episode {episode:3d} | "
                  f"Epsilon {self.epsilon:6.2f} | "
                  f"Memory Size {len(self.memory):5d} | "
                  f"Episode Return {episode_rewards:.2e} | "
                  f"Evaluation Score {validation_score:.2e} | "
                  f"Step {self.step_count} ")


            # update reward list of all episodes
            self.episode_returns.append(episode_rewards)
        # ------------- END loop 1

        print("...Training completed !")
        return self.episode_returns


    def normalize_state(self, state):
        """
        Normalize a given state using the precomputed mean and std.
        """
        return (state - self.state_mean) / self.state_std

    def save_full_state(self, path):
        """
        Save the full state of the agent, including the model, optimizer,
        replay buffer, epsilon, and step count.
        """

        # Clean None values from memory
        valid_memory = [x for x in self.memory.data if x is not None]

        state = {
            'model_state': self.model.state_dict(),
            'target_model_state': self.target_model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'memory': valid_memory,
            'memory_index': self.memory.index,
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'previous_val' : self.previous_val,
            'episode_returns' : self.episode_returns
        }

        torch.save(state, path)
        print(f"Agent's full state saved to {path}")

    def load_full_state(self, path):
        """
        Load the full state of the agent, including the model, optimizer,
        replay buffer, epsilon, and step count.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        state = torch.load(path, map_location=self.device, weights_only=False)

        # Load model weights
        self.model.load_state_dict(state['model_state'])
        self.target_model.load_state_dict(state['target_model_state'])

        # Load optimize
        self.optimizer.load_state_dict(state['optimizer_state'])

        # Load Replay Buffer
        self.memory.data = state['memory']
        self.memory.index = state['memory_index']

        # Load epsilon and step_count
        self.epsilon = state['epsilon']
        self.step_count = state['step_count']

        # Load previous_val and epsilon_returs
        self.previous_val = state['previous_val']
        self.episode_returns = state['episode_returns']

        print(f"Agent's full state loaded from {path}")

    def save(self,path):
        """Save model parameters to file."""
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")


    def load(self):
        """
        Load model parameters on CPU from file.
        We load on CPU because eval method will load the model on CPU.
        """
        import os

        # Get the directory where this script resides
        #script_dir = os.getcwd()                                #### for Notebooks ####
        script_dir = os.path.dirname(os.path.abspath(__file__))  ####  for file.py  ####

        # Construct the path to the model file
        path = os.path.join(script_dir, os.pardir, 'models', 'simpleDQN_DR_1.66e10.pt') #### change the name ####

        # Check if the file exists
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}. Please ensure the model file exists.")

        # Define the device as CPU
        device = torch.device("cpu")

        # Load model parameters and move to CPU
        self.model.load_state_dict(torch.load(path, map_location=device,weights_only=False))
        self.model.to(device)  # Ensure the model is on CPU
        self.model.eval()      # Set the model to evaluation mode

        # Synchronize the target network with the main model
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.to(device)
        self.target_model.eval()
        print(f"Model loaded from {path}")

#**************************************************
# Environment setup
env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200)

PATH = ""

# Configuration dictionary
CONFIG = {
    'input_dim': env.observation_space.shape[0], # Dimension of state space
    'output_dim' : env.action_space.n,           # Dimension of action space
    'gamma': 0.98,
    'learning_rate': 0.001,
    'nb_gradient_steps': 3,
    'target_update_steps': 400,
    'buffer_size': 100000,
    'batch_size': 790,
    'epsilon_max': 1.,          # Inital exploration rate.
    'epsilon_min': 0.02,        # Minimum value of epsilon.
    'epsilon_step': 5e-5,       # Decrement for epsilon at each step (for linear decay).
    'epsilon_delay_decay': 100, # Number of steps before starting epsilon decay.
    'save_path' : PATH+'models/simpleDQN_3.pt',
    'full_state_path' : PATH+'models/full_state_simpleDQN_3.pt',
}

# Instantiate the agent
#agent = ProjectAgent()

# Resume training if necessary
#agent.load()

# Train the agent
#num_episodes = 100 #500
#episode_returns = agent.train(env, num_episodes)

# Save the trained model
#agent.save(PATH+"models/simpleDQN.pt")
#**************************************************

