import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from typing import Dict, List, Any
from experiments.logger import setup_logger
from experiments.monteCarloTreeSearch_intergration import StateTransitionGraph, MCTS
import os
import pickle

class DQN(nn.Module):
    def __init__(self, state_dim: int, action_vector_dim: int, latent_dim: int = 128):
        super(DQN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        self.attention = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=8)
        self.network = nn.Sequential(
            nn.Linear(latent_dim + action_vector_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        encoded_state = self.encoder(state)  # [batch_size, latent_dim]
        encoded_state = encoded_state.unsqueeze(0)  # [1, batch_size, latent_dim]
        attn_output, _ = self.attention(encoded_state, encoded_state, encoded_state)
        encoded_state = attn_output.squeeze(0)  # [batch_size, latent_dim]
        
        if action.dim() == 1:
            action = action.unsqueeze(0).expand(encoded_state.size(0), -1)
        
        x = torch.cat((encoded_state, action), dim=-1)
        return self.network(x)

class DQNAgent:
    def __init__(self, state_dim: int, action_dim: int, log_dir: str = "app/logs/",
                 emulator_name: str = "Unknown", app_name: str = "Unknown"):
        """Initialize the DQN agent with state and action dimensions."""
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.logger = setup_logger(f"{log_dir}/dqn_agent.log", emulator_name=emulator_name, app_name=app_name)
        self.logger.info(f"DQNAgent initialized with state_dim: {state_dim}, action_dim: {action_dim}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize DQN networks with latent_dim
        self.policy_net = DQN(state_dim, action_dim, latent_dim=128).to(self.device)
        self.target_net = DQN(state_dim, action_dim, latent_dim=128).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer and loss
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0001)
        self.loss_fn = nn.MSELoss()
        
        # Experience replay
        self.memory = deque(maxlen=10000)

        self.stgraph = StateTransitionGraph(log_dir, emulator_name, app_name)
        self.mcts = MCTS(self, self.stgraph, num_simulations=100, c1=1.0, c2=1.0,
                         log_dir=log_dir, emulator_name=emulator_name, app_name=app_name)
        
        # Hyperparameters
        self.gamma = 1  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.target_update = 5  # Steps to update target network
        self.step_count = 0

    def get_q_values(self, state: torch.Tensor, action_vectors: List[torch.Tensor]) -> torch.Tensor:
        state = state.to(self.device).unsqueeze(0).expand(len(action_vectors), -1)
        actions = torch.stack(action_vectors).to(self.device)
        q_values = self.policy_net(state, actions)
        return q_values.squeeze()

    def act(self, state: torch.Tensor, actions: List[Dict[str, Any]], action_vectors: List[torch.Tensor]) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, len(actions) - 1)
        q_values = self.get_q_values(state, action_vectors)
        return q_values.argmax().item()

    def store(self, state: torch.Tensor, action_idx: int, reward: float, next_state: torch.Tensor, done: bool, action: Dict[str, Any] = None):
        """Store transition in memory and update graph."""
        state = state.to(self.device)
        next_state = next_state.to(self.device) if next_state is not None else None
        self.memory.append((state, action_idx, reward, next_state, done, action))
        if next_state is not None:
            self.stgraph.add_transition(state, action_idx, next_state, action)
            self.logger.debug(f"Stored transition: action {action_idx}, reward {reward}, done {done}, has next_state")
        else:
            self.logger.debug(f"Stored transition: action {action_idx}, reward {reward}, done {done}, no next_state")

    def train(self):
        """Train the DQN agent using a batch from replay memory."""
        if len(self.memory) < self.batch_size:
            self.logger.debug("Not enough experiences for training")
            return

        self.step_count += 1
        
        valid_transitions = [
            t for t in random.sample(self.memory, min(len(self.memory), self.batch_size))
            if t[1] >= 0 and t[3] is not None
        ]
        if len(valid_transitions) < self.batch_size:
            self.logger.debug(f"Not enough valid transitions for training: {len(valid_transitions)}/{self.batch_size}")
            return
        
        batch = valid_transitions
        states, actions, rewards, next_states, dones, _ = zip(*batch)
        
        states = torch.stack(states).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        
        q_values = self.policy_net(states, actions).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_q_values = self.target_net(next_states, actions).max(1)[0]
            targets = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = self.loss_fn(q_values, targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        if self.step_count % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.logger.info("Updated target network")
        
        self.logger.debug(f"Training step {self.step_count}: Loss={loss.item():.4f}, Epsilon={self.epsilon:.4f}")

    def save_model(self, path: str):
        """Save the policy network weights."""
        torch.save(self.policy_net.state_dict(), path)
        self.logger.info(f"Saved model to {path}")

    def load_model(self, path: str):
        """Load the policy network weights."""
        try:
            self.policy_net.load_state_dict(torch.load(path))
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.logger.info(f"Loaded model from {path}")
        except Exception as e:
            self.logger.error(f"Failed to load model from {path}: {str(e)}")

    def save_replay_buffer(self, path: str, reward_threshold: float = 500.0):
        try:
            high_value = [t for t in self.memory if t[2] > reward_threshold or (t[5] and t[5].get('parameters', {}).get('is_priority', False))]
            with open(path, 'wb') as f:
                pickle.dump(high_value, f)
            self.logger.info(f"Saved {len(high_value)} high-value transitions to {path}")
        except Exception as e:
            self.logger.error(f"Failed to save replay buffer: {str(e)}")

    def load_replay_buffer(self, path: str):
        """Load the replay buffer from a file."""
        try:
            with open(path, 'rb') as f:
                buffer = pickle.load(f)
            self.memory = deque(buffer[-self.memory.maxlen:], maxlen=self.memory.maxlen)
            self.logger.info(f"Loaded {len(self.memory)} transitions from {path}")
        except Exception as e:
            self.logger.error(f"Failed to load replay buffer from {path}: {str(e)}")