import os
import pickle
import random
from collections import deque
from typing import Dict, List, Any

import torch
import torch.nn as nn
import torch.optim as optim

from experiments.logger import setup_logger
from experiments.monteCarloTreeSearch_intergration import StateTransitionGraph, MCTS
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
import hashlib
from experiments.utils.prioritized_replay_buffer import PrioritizedReplayBuffer # Import PER

def hash_tensor(t: torch.Tensor) -> str:
    return hashlib.md5(t.cpu().numpy().tobytes()).hexdigest()


class MacroManager:
    def __init__(self, sequence_generator, state_graph):
        self.seq_gen = sequence_generator  # LSTM decoder
        self.graph = state_graph           # StateTransitionGraph
        self.current_macro = []

    def reset(self):
        self.current_macro.clear()

    def get_next_macro_action(self, state, state_hash, actions, action_vectors):
        if self.current_macro:
            return self.current_macro.pop(0)

        # --- Graph motif match ---
        graph_match = self.graph.match_motif(state_hash)
        if graph_match:
            matched_action = eval(graph_match)
            self.current_macro = [matched_action]
            return self.current_macro.pop(0)

        # --- Sequence model prediction ---
        self.current_macro = self.seq_gen.generate_macro(state, actions, action_vectors)
        if self.current_macro:
            return self.current_macro.pop(0)

        return None


class MacroSequenceDataset(Dataset):
    def __init__(self, replay_buffer, max_seq_len=5):
        self.samples = []
        self.max_seq_len = max_seq_len
        self._prepare_data(replay_buffer)

    def _prepare_data(self, buffer):
        sequence = []
        # Iterate through the buffer to extract sequences
        # If using PrioritizedReplayBuffer, we need to get the actual experiences
        # This assumes the buffer stores (state, action_idx, reward, next_state, done, action_vectors)
        # If buffer is PER, it stores (priority, (state, ...))
        
        # For PER, we need to extract the actual experiences from the SumTree
        # This is a simplified way to get all experiences, not efficient for large buffers
        if isinstance(buffer, PrioritizedReplayBuffer):
            all_experiences = []
            for i in range(buffer.tree.n_entries):
                _, _, experience = buffer.tree.get(buffer.tree.total() * (i / buffer.tree.n_entries))
                all_experiences.append(experience)
        else:
            all_experiences = buffer # Assume it's a deque

        for i in range(len(all_experiences)):
            state, _, _, _, done, action_vecs = all_experiences[i]
            if action_vecs:
                sequence.append((state, action_vecs))
            if done or len(sequence) >= self.max_seq_len:
                if len(sequence) > 1:
                    input_state = sequence[0][0]
                    target_actions = [x[1][0] for x in sequence[:self.max_seq_len]]
                    self.samples.append((input_state, target_actions))
                sequence = []

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        state, targets = self.samples[idx]
        targets = torch.stack(targets)
        return state, targets
    

def train_macro_generator( replay_buffer, state_dim, action_dim, device='cuda', epochs=5, batch_size=8):
    dataset = MacroSequenceDataset(replay_buffer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


    model = MacroSequenceGenerator(state_dim, action_dim).to(device)
    # load 
    model.load_state_dict(torch.load("macro_gen.pth", map_location=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0
        for states, target_seq in dataloader:
            states = states.to(device)
            target_seq = target_seq.to(device)

            pred_seq = model(states)
            loss = loss_fn(pred_seq, target_seq)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss = {total_loss / len(dataloader):.4f}")

    torch.save(model.state_dict(), "macro_gen.pth")
    return model


class MacroSequenceGenerator(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, max_seq_len=5):
        super().__init__()
        self.encoder = nn.Linear(state_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.max_seq_len = max_seq_len

    def forward(self, state_embedding):
        batch_size = state_embedding.size(0)
        x = self.encoder(state_embedding).unsqueeze(1)
        x = x.expand(-1, self.max_seq_len, -1)  # Repeat for sequence length
        lstm_out, _ = self.lstm(x)
        return self.action_head(lstm_out)  # [batch, seq_len, action_dim]
    
    def generate_macro(self, state: torch.Tensor, actions: list, action_vectors: list, max_len: int = 5):
        self.eval()
        with torch.no_grad():
            state_input = state.unsqueeze(0).to(next(self.parameters()).device)  # [1, state_dim]
            x = self.encoder(state_input).unsqueeze(1).expand(-1, max_len, -1)   # [1, max_len, hidden]
            lstm_out, _ = self.lstm(x)
            predictions = self.action_head(lstm_out).squeeze(0)  # [max_len, action_dim]

        # Choose the most similar real action at each step
        macro_actions = []
        av_tensor = torch.stack(action_vectors).to(predictions.device)  # [num_actions, action_dim]
        for step_vec in predictions:
            similarities = torch.matmul(av_tensor, step_vec)
            best_idx = torch.argmax(similarities).item()
            macro_actions.append(actions[best_idx])

        return macro_actions



class ICM(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128):
        super().__init__()
        self.forward_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, state_dim)
        )

    def forward(self, state, action, next_state):
        device = next(self.parameters()).device  # Ensure consistent device
        state = state.to(device)
        action = action.to(device)
        next_state = next_state.to(device)

        input_vector = torch.cat([state, action], dim=-1)
        predicted_next = self.forward_model(input_vector)
        loss = F.mse_loss(predicted_next, next_state)
        return loss


# ===== Proposed Dueling Double DQN Architecture =====
class DuelingDQN(nn.Module):
    def __init__(self, state_dim: int, action_vector_dim: int, latent_dim: int = 128):
        super(DuelingDQN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim,256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
            nn.ReLU()
        )
        self.value_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.advantage_head = nn.Sequential(
            nn.Linear(latent_dim + action_vector_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.lstm = nn.LSTM(latent_dim, latent_dim, batch_first=True)

    def forward(self, state: torch.Tensor, actions: torch.Tensor, hidden_state=None):
        z = self.encoder(state)  # [batch, latent_dim]

        # Add a dummy time dimension for LSTM: [batch, seq_len=1, latent_dim]
        z_seq = z.unsqueeze(1)
        z_lstm_out, hidden = self.lstm(z_seq, hidden_state)  # z_lstm_out: [batch, 1, latent_dim]
        z = z_lstm_out.squeeze(1)  # Remove sequence dim

        # Continue as before with the advantage stream
        num_actions = actions.size(1)
        z_expanded = z.unsqueeze(1).expand(-1, num_actions, -1)
        x = torch.cat((z_expanded, actions), dim=-1)
        a = self.advantage_head(x).squeeze(-1)
        v = self.value_head(z)

        return v + (a - a.mean(dim=1, keepdim=True)), hidden

class DQNAgent:
    def __init__(self, state_dim: int, action_vector_dim: int, log_dir: str = "app/logs/",
                emulator_name: str = "Unknown", app_name: str = "Unknown"):
        self.state_dim = state_dim
        self.action_vector_dim = action_vector_dim
        self.hidden_state = None

        self.logger = setup_logger(f"{log_dir}/dqn_agent_improved.log", emulator_name=emulator_name, app_name=app_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # replace DQN with DuelingDQN

        self.logger.info(f"DQNAgent initialized with state_dim: {state_dim}, action_vector_dim: {action_vector_dim}")

        self.policy_net = DuelingDQN(state_dim, action_vector_dim).to(self.device)
        self.target_net = DuelingDQN(state_dim, action_vector_dim).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)
        self.loss_fn = nn.MSELoss()


        self.stgraph = StateTransitionGraph(log_dir, emulator_name, app_name)
        self.mcts = MCTS(self, self.stgraph, num_simulations=100, c1=1.0, c2=1.0,
                         log_dir=log_dir, emulator_name=emulator_name, app_name=app_name)
        

        self.icm = ICM(state_dim, action_vector_dim).to(self.device)
        self.beta = 0.05  # Scale for intrinsic reward
        self.current_macro = []  # Macro action queue

        self.macro_generator = MacroSequenceGenerator(state_dim, action_vector_dim).to(self.device)
        #self.macro_generator.load_state_dict(torch.load(f"{log_dir}/macro_gen.pth", map_location=self.device))
        self.macro_manager = MacroManager(sequence_generator=self.macro_generator, state_graph=self.stgraph)


        # Prioritized Replay Buffer
        self.memory = PrioritizedReplayBuffer(capacity=10000)

       
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 16 
        self.target_update = 100
        self.step_count = 0


    # def act(self, state: torch.Tensor, actions: List[Dict[str, Any]], action_vectors: List[torch.Tensor]) -> int:
        # if random.random() < self.epsilon:
        #     return random.randrange(len(actions))
        # sv = state.to(self.device).unsqueeze(0)  # [1, state_dim]
        # av = torch.stack(action_vectors).to(self.device).unsqueeze(0)  # [1, num_actions, action_vector_dim]
        # qvals = self.policy_net(sv, av).squeeze(0)  # [num_actions]
        # return torch.argmax(qvals).item()
        
    def action_possiblity_calc(self, action_vector: torch.Tensor, state_vector: torch.Tensor, actions: list[Dict]) -> int:
        av = action_vector.to(self.device).unsqueeze(0)
        sv = state_vector.to(self.device).unsqueeze(0)
        
        av = torch.stack(action_vectors).to(self.device).unsqueeze(0)

        state_hash = hash_tensor(state)

        with torch.no_grad():
            qvals, self.hidden_state = self.policy_net(sv, av, self.hidden_state)

        if random.random() < 0.2:
            macro_action = self.macro_manager.get_next_macro_action(state, state_hash, actions, action_vectors)
            if macro_action:
                return next((i for i, a in enumerate(actions) if a == macro_action), random.randint(0, len(actions) - 1))

        return torch.argmax(qvals.squeeze(0)).item()

    
    def store(self, state: torch.Tensor, action_idx: int, reward: float, next_state: torch.Tensor, done: bool, action: Dict[str, Any] = None):
        """Store transition in memory and update graph."""
        state = state.to(self.device)
        next_state = next_state.to(self.device) if next_state is not None else torch.zeros(self.state_dim, device=self.device) # Ensure next_state is a tensor
        # self.logger.info(f"State: {state}")
        # self.logger.info(f"ActionIdx: {action_idx}")
        # self.logger.info(f"Reward: {reward}")
        # self.logger.info(f"Next State: {next_state}")
        # self.logger.info(f"Done: {done}")
        # self.logger.info(f"Action: {action}")

        #print(f"State: {state}, Action: {action_idx}, Reward: {reward}, Next State: {next_state}, Done: {done}")
        reward = reward/1000.0
        # Calculate initial priority (e.g., max priority or a small positive value)
        max_p = self.memory.tree.total() if self.memory.tree.n_entries > 0 else 1.0
        self.memory.add(max_p, (state, action_idx, reward, next_state, done, action))
        

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        self.step_count += 1
        
        # Sample from PER buffer
        batch_data = self.memory.sample(self.batch_size)
        indices, experiences = zip(*batch_data)
        states, actions, rewards, next_states, dones, next_action_vectors = zip(*experiences)

        states = torch.stack(states).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.stack(next_states).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # self.logger.info(f"States: {states.shape}")
        # self.logger.info(f"Actions: {actions.shape}")
        # self.logger.info(f"Rewards: {rewards.shape}")
        # self.logger.info(f"Next States: {next_states.shape}")
        # self.logger.info(f"Dones: {dones.shape}")
        
        av_batch = []
        for nav in next_action_vectors:
            if nav is None:
                # Use a zero tensor with the same shape as other action vectors
                default_action = torch.zeros(self.action_vector_dim, device=self.device)
                av_batch.append(default_action.unsqueeze(0))
            else:
                av_batch.append(torch.stack(nav).to(self.device))

        # Compute current Q-values
        max_actions = max(len(nav) for nav in av_batch)
        av_padded = torch.zeros(self.batch_size, max_actions, av_batch[0].size(1), device=self.device)
        for i, av in enumerate(av_batch):
            av_padded[i, :len(av)] = av
            #print(f"av_padded[{i}]: {len(av_padded[i])}")
        
        q_current, hidden = self.policy_net(states, av_padded)  # [batch, max_actions]
        # self.logger.debug(f"Hidden state: {hidden}")
        # self.logger.debug(f"Q_current shape: {q_current.shape}")

        q_current = q_current[range(self.batch_size), actions]  # Select Q-values for taken actions

        # Compute target Q-values
        with torch.no_grad():
            q_next_policy, _ = self.policy_net(next_states, av_padded)  # [batch, max_actions]
            best_next = torch.argmax(q_next_policy, dim=1)  # [batch]
            q_next_target, _ = self.target_net(next_states, av_padded)  # [batch, max_actions]
            q_next_best = q_next_target[range(self.batch_size), best_next]
            target = rewards + (1 - dones) * self.gamma * q_next_best
            # self.logger.debug(f"Q_current: {q_current}")
            # self.logger.debug(f"Target Q: {target}")

        # Compute Q-learning loss
        loss = self.loss_fn(q_current, target)

        # Compute TD-error for PER update
        td_error = torch.abs(target - q_current).cpu().numpy()
        for i, idx in enumerate(indices):
            self.memory.update(idx, td_error[i])

        # Compute ICM loss BEFORE backward
        icm_loss_total = 0
        for i in range(self.batch_size):
            if i < len(av_batch) and actions[i] < len(av_batch[i]):
                intrinsic_loss = self.icm(states[i], av_batch[i][actions[i]], next_states[i])
                icm_loss_total += intrinsic_loss

        icm_loss_avg = icm_loss_total / self.batch_size

        # Combine both into a single graph-safe loss
        total_loss = loss + self.beta * icm_loss_avg
        self.logger.warning(f"Loss: {loss.item()}, ICM Loss: {icm_loss_avg.item()}, Total Loss: {total_loss.item()}")
        # One backward pass only
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()


    def save_model(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.policy_net.state_dict(), path)
        self.logger.info(f"Model saved: {path}")

    def load_model(self, path: str):
        if not os.path.exists(path):
            self.logger.error(f"No model at {path}")
            return
        try:
            state_dict = torch.load(path, map_location=self.device)
            self.policy_net.load_state_dict(state_dict)
            self.target_net.load_state_dict(state_dict)
            self.logger.info(f"Model loaded: {path}")
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")

    def save_replay_buffer(self, path: str, reward_threshold: float = 500.0):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            # For PER, we save the entire SumTree structure or reconstruct it from data
            # For simplicity, let\'s save the raw experiences for now.
            # A more robust solution would serialize the SumTree.
            data = []
            for i in range(self.memory.tree.n_entries):
                _, _, experience = self.memory.tree.get(self.memory.tree.total() * (i / self.memory.tree.n_entries))
                data.append(experience)
            
            # Filter by reward threshold if needed, but for PER, all experiences are valuable
            # data = [t for t in data if t[2] > reward_threshold]

            with open(path, 'wb') as f:
                pickle.dump(data, f)
            self.logger.info(f"Replay buffer saved: {path} with {len(data)} entries.")
        except Exception as e:
            self.logger.error(f"Error saving replay buffer: {e}")

    def load_replay_buffer(self, path: str):
        if not os.path.exists(path):
            self.logger.error(f"No replay buffer at {path}")
            return
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            
            # Reconstruct PER buffer from loaded data
            self.memory = PrioritizedReplayBuffer(capacity=10000) # Re-initialize
            for experience in data:
                # Add with a default high priority, or re-calculate if possible
                self.memory.add(1.0, experience) # Add with max priority
            
            self.logger.info(f"Replay buffer loaded: {path} with {len(data)} entries.")
        except Exception as e:
            self.logger.error(f"Error loading replay buffer: {e}")



