import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from typing import Dict, List, Any, Tuple
from experiments.logger import setup_logger

# ===== Proposed Dueling Double DQN Architecture =====
class DuelingDQN(nn.Module):
    def __init__(self, state_dim: int, action_vector_dim: int, latent_dim: int = 128):
        super(DuelingDQN, self).__init__()
        # shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
            nn.ReLU()
        )
        # value stream
        self.value_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        # advantage stream (takes concatenated [encoded_state, action_vector])
        self.advantage_head = nn.Sequential(
            nn.Linear(latent_dim + action_vector_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # encode state
        z = self.encoder(state)                           # [batch, latent_dim]
        # value
        v = self.value_head(z)                            # [batch, 1]
        # prepare action
        if action.dim() == 1:
            action = action.unsqueeze(0).expand(z.size(0), -1)
        # advantage
        x = torch.cat((z, action), dim=-1)
        a = self.advantage_head(x)                        # [batch, 1]
        # combine: Q = V + (A - mean(A))
        return v + (a - a.mean(dim=0, keepdim=True))

class DQNAgentImproved:
    def __init__(self, state_dim: int, action_dim: int, log_dir: str = "app/logs/",
                 emulator_name: str = "Unknown", app_name: str = "Unknown"):
        self.logger = setup_logger(f"{log_dir}/dqn_agent_improved.log", emulator_name=emulator_name, app_name=app_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # replace DQN with DuelingDQN
        self.policy_net = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_net = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)
        self.loss_fn = nn.MSELoss()

        # prioritized buffer stub (can be upgraded)
        self.memory = deque(maxlen=10000)

        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.target_update = 1000
        self.step_count = 0

    def act(self, state: torch.Tensor, actions: List[Dict[str, Any]], action_vectors: List[torch.Tensor]) -> int:
        if random.random() < self.epsilon:
            return random.randrange(len(actions))
        # evaluate all Q-values
        sv = state.to(self.device).unsqueeze(0).expand(len(action_vectors), -1)
        av = torch.stack(action_vectors).to(self.device)
        qvals = self.policy_net(sv, av).squeeze()
        return torch.argmax(qvals).item()

    def store(self, state: torch.Tensor, action_idx: int, reward: float, next_state: torch.Tensor, done: bool, action: Dict[str, Any] = None):
        """Store transition in memory and update graph."""
        state = state.to(self.device)
        next_state = next_state.to(self.device) if next_state is not None else None
        self.memory.append((state, action_idx, reward, next_state, done, action))
       

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        self.step_count += 1
        # sample
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones, next_action_vectors = zip(*batch)

        # to tensors
        states = torch.stack(states).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.stack(next_states).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        naps = torch.stack(next_action_vectors).to(self.device)

        # current Q
        q_current = self.policy_net(states, torch.stack([av for av in naps])).squeeze()
        q_current = q_current.gather(0, actions.unsqueeze(1)).squeeze()

        # Double DQN target: select best action via policy, evaluate via target
        with torch.no_grad():
            # get next Q values for all next actions
            q_next_policy = self.policy_net(next_states, naps).squeeze()
            best_next = torch.argmax(q_next_policy, dim=0)
            q_next_target = self.target_net(next_states, naps).squeeze()
            q_next_best = q_next_target.gather(0, best_next.unsqueeze(1)).squeeze()
            target = rewards + (1 - dones) * self.gamma * q_next_best

        loss = self.loss_fn(q_current, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # epsilon decay
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # update target periodically
        if self.step_count % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.logger.info("Updated target network")
