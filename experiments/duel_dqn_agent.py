import sys
import traceback
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import random
from collections import deque
import torch.optim as optim
from experiments.logger import setup_logger
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DenseBlock(nn.Module):
    """
    Fully-connected Dense Block.
    Each layer's output is concatenated with all previous outputs and the original input.
    """

    def __init__(self, input_dim, growth_rate=64, num_layers=4, use_bn=False):
        super().__init__()
        self.layers = nn.ModuleList()
        hidden_dim = growth_rate

        for i in range(num_layers):
            in_features = input_dim + i * growth_rate
            block = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.BatchNorm1d(hidden_dim) if use_bn else nn.Identity(),
                nn.ReLU(inplace=True),
            )
            self.layers.append(block)

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            concatenated = torch.cat(features, dim=1)
            out = layer(concatenated)
            features.append(out)
        return torch.cat(features, dim=1)

class MetaController(nn.Module):
    def __init__(self, input_dim=96, hidden_dim=64, num_modes=4):
        super().__init__()
        self.feature = DenseBlock(input_dim, growth_rate=hidden_dim, num_layers=2)
        feature_out_dim = input_dim + 2 * hidden_dim  # 2 layers
        self.out = nn.Linear(feature_out_dim, num_modes)

    def forward(self, state_vec):
        # state_vec: [B, 96]
        features = self.feature(state_vec)
        return self.out(features)  # logits over modes


class DuelingDQN(nn.Module):
    def __init__(
        self,
        log_func,
        state_tensor_dim,
        action_space_tensor_dim,
        action_tensor_dim,
        growth_rate=16,
        num_layers=4,
    ):
        super().__init__()
        self.logger = log_func

        combined_dim = state_tensor_dim + action_space_tensor_dim
        feature_dim = combined_dim + num_layers * growth_rate
        # DenseBlock produces [n, feature_dim]
        self.feature = DenseBlock(
            combined_dim, growth_rate=growth_rate, num_layers=num_layers
        )

        self.estimator_out = 32
        # State‐value head: takes feature + state
        self.estimator1 = nn.Sequential(
            nn.Linear(feature_dim + state_tensor_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, self.estimator_out),
        )
        # Advantage head: feature + action
        self.estimator2 = nn.Sequential(
            nn.Linear(feature_dim + action_space_tensor_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, self.estimator_out),
        )

        # Q‐value: concat action one‐hot + two heads
        q_input_dim = action_tensor_dim + 2 * self.estimator_out
        self.q_val_calc = nn.Sequential(nn.Linear(q_input_dim, 1))

    def forward(self, state_tensor, action_space_tensor, actions: torch.Tensor):
        try:
            print("actions.shape:", actions.shape)
            # state_tensor: [1, S], action_space_tensor: [n, A], actions: [n, action_dim]
            n = action_space_tensor.size(0)
            state_expanded = state_tensor.expand(n, -1)  # [n, S]
            combined = torch.cat((state_expanded, action_space_tensor), dim=1)  # [n, S+A]
            print("combined.shape:", combined.shape)
            features = self.feature(combined)  # [n, feature_dim]
            print("features.shape:", features.shape)
            # State value head
            v_input = torch.cat((features, state_expanded), dim=1)  # [n, feature+S]
            print("estimator1_input (v_input).shape:", v_input.shape)
            v = self.estimator1(v_input)  # [n, estimator_out]
            print("v.shape:", v.shape)
            # Advantage head
            a_input = torch.cat((features, action_space_tensor), dim=1)  # [n, feature+A]
            print("estimator2_input (a_input).shape:", a_input.shape)
            a = self.estimator2(a_input)  # [n, estimator_out]
            print("a.shape:", a.shape)
            # Combine with one‐hot action encoding
            q_input = torch.cat(
                (actions, v, a), dim=1
            )  # [n, action_tensor_dim + 2*estimator_out]
            print("q_input.shape:", q_input.shape)
            q = self.q_val_calc(q_input).squeeze(1)  # [n]

            return q
        except Exception as e:
            self.logger.error(traceback.print_exc())
            return torch.zeros(1)


class EventSelector:
    def __init__(
        self, log_func, tau=0.03, epsilon_start=0.9, epsilon_end=0.3, epsilon_steps=1000
    ):
        self.logger = log_func
        self.tau = tau  # lower tau, pay more attention to the UI
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_steps = epsilon_steps
        self.step = 0

    def get_epsilon(self):
        if self.step >= self.epsilon_steps:
            return self.epsilon_end
        epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * (
            self.step / self.epsilon_steps
        )
        self.step += 1
        return epsilon

    def select_event(
        self,
        dqn,
        state_tensor: torch.Tensor,
        action_tensors: torch.Tensor,
        actions: torch.Tensor,
    ):
        # action_tensors [n, 86]
        # state_tensor[1,96]
        # actions [n, 13]
        epsilon = self.get_epsilon()
        if random.random() < epsilon:
            idx = random.choice(range(actions.size(0)))
            return idx
        else:
            with torch.no_grad():
                self.logger.info(f"Shape of state_tensor: {state_tensor.shape}")
                self.logger.info(f"Shape of action_tensors: {action_tensors.shape}")

                actions_for_model = actions[:, :-1]  # strip index
                self.logger.info(f"Shape of actions: {actions_for_model.shape}")
                q_values = dqn(
                    state_tensor.to(device),
                    action_tensors.to(device),
                    actions_for_model.to(device),
                )
                self.logger.info(
                    f"Q-values: {q_values} actions {actions.cpu().numpy()}, epsilon: {epsilon}"
                )
                idx = q_values.argmax().item()
                return idx


class PrioritizedReplayBuffer:
    def __init__(self, log_func, capacity, alpha=0.6):
        self.logger = log_func
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)  # base on td_error => predict Q-value is far-off from target Q-value
        self.alpha = alpha

    def push(
        self,
        state,
        widget_action_state,
        action,
        reward,
        next_state,
        next_widget_action_state,
        done,
    ):
        max_priority = max(self.priorities) if self.priorities else 1.0
        self.buffer.append(
            (
                state,
                widget_action_state,
                action,
                reward,
                next_state,
                next_widget_action_state,
                done,
            )
        )

        def info(x, name=""):
            if isinstance(x, torch.Tensor):
                summary = f"Tensor, shape={tuple(x.shape)}, dtype={x.dtype}"
                val = (
                    x.detach().cpu().numpy()
                    if x.numel() <= 10
                    else f"{x.detach().cpu().numpy().flatten()[:5]}..."
                )
            elif isinstance(x, np.ndarray):
                summary = f"ndarray, shape={x.shape}, dtype={x.dtype}"
                val = x if x.size <= 10 else f"{x.flatten()[:5]}..."
            elif isinstance(x, (list, tuple)):
                summary = f"{type(x).__name__}, len={len(x)}"
                val = x if len(x) <= 10 else f"{x[:5]}..."
            else:
                summary = f"{type(x).__name__}"
                val = x
            return f"{name}=[{summary}] → {val}"

        # Usage
        self.logger.warning(
            info(state, "state")
            + ", "
            + info(widget_action_state, "widget_act")
            + ", "
            + info(action, "action")
            + ", "
            + info(reward, "reward")
            + ", "
            + info(next_state, "next_state")
            + ", "
            + info(next_widget_action_state, "next_widget_act")
            + ", "
            + info(done, "done")
        )

        self.priorities.append(max_priority)

    def sample(self, batch_size, beta=0.4):
        try:
            if len(self.buffer) == 0:
                return None
            priorities = np.array(self.priorities, dtype=np.float32)
            probs = priorities**self.alpha
            probs /= probs.sum()
            indices = np.random.choice(len(self.buffer), batch_size, p=probs)
            samples = [self.buffer[idx] for idx in indices]
            weights = (len(self.buffer) * probs[indices]) ** (-beta)
            weights /= weights.max()
            weights = torch.FloatTensor(weights).to(device)
            (
                states,
                widget_action_state,
                actions,
                rewards,
                next_states,
                next_widget_action_state,
                dones,
            ) = zip(*samples)
            return (
                torch.stack(states).to(device),
                torch.stack(widget_action_state).to(device),
                torch.stack([torch.tensor(a, dtype=torch.float32) for a in actions]).to(device),
                torch.tensor(rewards, dtype=torch.float32).to(device),
                torch.stack(next_states).to(device),
                torch.stack(next_widget_action_state).to(device),
                torch.tensor(dones, dtype=torch.float32).to(device),
                indices,
                weights,
            )
        except Exception as e:
            print(f"Error in sample: {e}")
            self.logger.error(traceback.print_exc())
            sys.exit(1)

    def export_to_csv(self, path="buffer_export.csv"):
        try:
            records = []
            for (
                state,
                widget_action_state,
                action,
                reward,
                next_state,
                next_widget_action_state,
                done,
            ) in self.buffer:
                records.append(
                    {
                        "state": str(
                            state.cpu().numpy().tolist()
                            if isinstance(state, torch.Tensor)
                            else state
                        ),
                        "widget_action_state": str(
                            widget_action_state.cpu().numpy().tolist()
                            if isinstance(widget_action_state, torch.Tensor)
                            else widget_action_state
                        ),
                        "action": str(
                            action.cpu().numpy().tolist()
                            if isinstance(action, torch.Tensor)
                            else action
                        ),
                        "reward": str(reward),
                        "next_state": str(
                            next_state.cpu().numpy().tolist()
                            if isinstance(next_state, torch.Tensor)
                            else next_state
                        ),
                        "next_widget_action_state": str(
                            next_widget_action_state.cpu().numpy().tolist()
                            if isinstance(next_widget_action_state, torch.Tensor)
                            else next_widget_action_state
                        ),
                        "done": str(done),
                    }
                )

            df = pd.DataFrame(records)
            df.to_csv(path, index=False)
            self.logger.info(f"Replay buffer exported to CSV: {path}")
        except Exception as e:
            self.logger.error(f"Error exporting replay buffer to CSV: {e}")
            self.logger.error(traceback.print_exc())

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-5

    def save(self, path):
        torch.save(
            {"buffer": list(self.buffer), "priorities": list(self.priorities)}, path
        )

    def load(self, path):
        checkpoint = torch.load(path)
        self.buffer = deque(checkpoint["buffer"], maxlen=self.capacity)
        self.priorities = deque(checkpoint["priorities"], maxlen=self.capacity)


class DQNAgent:  # Adapter
    def __init__(
        self,
        log_dir="app/logs",
        app_name="unk",
        emulator_name="unk",
        ver="1",
        state_tensor_dim=96,
        action_space_tensor_dim=86,
        action_dim=13,
        lr=1e-3,
        gamma=0.99,
        tau=0.005,
        buffer_capacity=10000,
        batch_size=32,
    ):
        self.log_dir = log_dir
        self.model_path = f"{self.log_dir}/dqn_model.pth"
        self.replay_buffer_path = f"{self.log_dir}/replay_buffer.pth"
        self.logger = setup_logger(
            f"{self.log_dir}/dqn_agent.log",
            emulator_name=emulator_name,
            app_name=app_name,
        )

        self.state_tensor_dim = state_tensor_dim
        self.action_space_tensor_dim = action_space_tensor_dim
        self.action_dim = action_dim

        self.gamma = gamma
        self.tau = tau

        self.batch_size = batch_size  # need to match episode_length

        self.online_dqn = DuelingDQN(
            log_func=self.logger,
            state_tensor_dim=state_tensor_dim,
            action_space_tensor_dim=self.action_space_tensor_dim,
            action_tensor_dim=self.action_dim,
        ).to(device)
        self.target_dqn = DuelingDQN(
            log_func=self.logger,
            state_tensor_dim=state_tensor_dim,
            action_space_tensor_dim=self.action_space_tensor_dim,
            action_tensor_dim=self.action_dim,
        ).to(device)

        self.update_target_model()  # Initialize target model with online model weights
        self.target_dqn.eval()

        self.optimizer = optim.Adam(self.online_dqn.parameters(), lr=lr)
        self.loss_fn =  nn.SmoothL1Loss()
        self.selector = EventSelector(self.logger,epsilon_steps=800)
        self.memory = PrioritizedReplayBuffer(self.logger, buffer_capacity)
        self.train_history = []  # Store history for analysis

    def update_target_model(
        self,
    ):  # Update target model weights with online model weights
        self.target_dqn.load_state_dict(self.online_dqn.state_dict())

    def select_action(
        self, state_vector_tensor, widget_actions_vector_tensor, possible_actions # Use when interacting with the environment
    ):
        action_dim = self.action_dim

        state_vector_tensor = state_vector_tensor.to(device)
        widget_actions_vector_tensor_expanded = []
        _action_vectors_1hot = []
        for idx, action_dict in enumerate(possible_actions):
            action_vec = action_dict.get("actions", [])
            corr_widget_actions_vector_tensor = widget_actions_vector_tensor[idx]

            for action_idx in range(action_dim):
                if action_idx in action_vec:
                    one_hot = [0] * (action_dim + 1)
                    # +1 for the index of the action in the action space
                    one_hot[-1] = idx  # last element is the action index
                    one_hot[action_idx] = 1
                    _action_vectors_1hot.append(
                        torch.tensor(one_hot, dtype=torch.float32)
                    )
                    widget_actions_vector_tensor_expanded.append(
                        corr_widget_actions_vector_tensor
                    )

        corr_widget_actions_vector_tensor_expanded_tensor = torch.stack(
            widget_actions_vector_tensor_expanded
        ).to(device)
        action_vectors_tensor = torch.stack(_action_vectors_1hot, dim=0)

        print(
            f"action_vectors_tensor.shape: {action_vectors_tensor.shape}, corr_widget_actions_vector_tensor_expanded_tensor.shape: {corr_widget_actions_vector_tensor_expanded_tensor.shape}"
        )

        selector_action_idx = self.selector.select_event(
            dqn=self.online_dqn,
            state_tensor=state_vector_tensor,
            action_tensors=corr_widget_actions_vector_tensor_expanded_tensor,
            actions=action_vectors_tensor,
        )

        widget_action_idx = _action_vectors_1hot[selector_action_idx][-1]
        _1hot_action_vec = _action_vectors_1hot[selector_action_idx][:-1]
        self.logger.info(
            f"Selected action index: {widget_action_idx}, widget action index: {_1hot_action_vec}-{len(_1hot_action_vec)}"
        )
        return int(widget_action_idx), _1hot_action_vec.tolist()

    def train_replay(self):
        if len(self.memory.buffer) < self.batch_size:
            return

        samples = self.memory.sample(self.batch_size)
        if samples is None:
            return

        (
            states,
            widget_action_state,
            actions,
            rewards,
            next_states,
            next_widget_action_state,
            dones,
            indices,
            weights,
        ) = samples

        # ===== Ensure all tensors are on the correct device =====
        states = states.to(device)
        widget_action_state = widget_action_state.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        next_states = next_states.to(device)
        next_widget_action_state = next_widget_action_state.to(device)
        dones = dones.to(device)
        weights = weights.to(device)

        with torch.no_grad():
            # 1. Compute Q-values for next states using online DQN
            next_q_online = self.online_dqn(next_states, next_widget_action_state, actions)  # [batch_size]
            # Find best actions from online network
            best_action_idxs = next_q_online.argmax(dim=0)  # index of best action per batch

            # 2. Evaluate using target DQN
            next_q_target_all = self.target_dqn(next_states, next_widget_action_state, actions)  # [batch_size]
            max_next_q = next_q_target_all[best_action_idxs].unsqueeze(0)  # gather best target Qs

            # 3. Bellman target
            targets = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * max_next_q


        current_q = self.online_dqn(states, widget_action_state, actions).to(device)
        if current_q.dim() == 1:
            current_q = current_q.unsqueeze(1)

        td_errors = (current_q - targets).pow(2).squeeze(1)
        loss = (weights * td_errors).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.logger.warning(f"Train Loss: {str(loss.cpu().item())} | Temporal Difference: {str(td_errors.mean().cpu().item())}")
        self.train_history.append((loss.cpu().item(), td_errors.mean().cpu().item()))
        # Optional gradient clipping:
        torch.nn.utils.clip_grad_norm_(self.online_dqn.parameters(), max_norm=10.0)
        self.optimizer.step()

        new_priorities = torch.abs(current_q - targets).squeeze(1).detach().cpu().numpy()
        self.memory.update_priorities(indices, new_priorities)

        self.soft_update()

    def save_training_history(self, path="train_history.csv"):
        try:
            df = pd.DataFrame(self.train_history, columns=["loss", "td_error"])
            df.to_csv(path, index=False)
            self.logger.info(f"Training history saved to {path}")
            df.plot()
            plt.title("TD Error vs Loss")
            plt.grid()
            plt.savefig(f"{self.log_dir}/train_history_plot.png")
        except Exception as e:
            self.logger.error(f"Error saving training history: {e}")


    def soft_update(self):
        for target_param, online_param in zip(
            self.target_dqn.parameters(), self.online_dqn.parameters()
        ):
            target_param.data.copy_(
                self.tau * online_param.data + (1.0 - self.tau) * target_param.data
            )

    def save_model(self, path):
        torch.save(self.online_dqn.state_dict(), path)

    def load_model(self, path):
        self.online_dqn.load_state_dict(torch.load(path))
        self.target_dqn.load_state_dict(self.online_dqn.state_dict())
        self.target_dqn.eval()

    def save_replay_buffer(self, path):
        self.memory.save(path)

    def load_replay_buffer(self, path):
        self.memory.load(path)

class MetaDQNAgent:
    def __init__(self, input_dim=96, num_modes=4, hidden_dim=64, lr=1e-3):
        self.policy = MetaController(input_dim, hidden_dim, num_modes).to(device)
        self.target = MetaController(input_dim, hidden_dim, num_modes).to(device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.loss_fn = nn.CrossEntropyLoss(reduction="none")  # used for weighting
        self.gamma = 0.99
        self.replay_buffer = PrioritizedReplayBuffer(lambda x: x, capacity=5000)

    def select_mode(self, state_vec, epsilon=0.1):
        if random.random() < epsilon:
            return random.randint(0, self.policy.out.out_features - 1)
        with torch.no_grad():
            logits = self.policy(state_vec)
            return torch.argmax(logits, dim=-1).item()

    def train_step(self, batch_size=32):
        if len(self.replay_buffer.buffer) < batch_size:
            return

        (states, _, actions, rewards, next_states, _, dones, indices, weights) = \
            self.replay_buffer.sample(batch_size)

        states = states.to(device)
        next_states = next_states.to(device)
        actions = actions.to(device).long()  # class labels
        rewards = rewards.to(device)
        dones = dones.to(device)
        weights = weights.to(device)

        logits = self.policy(states)
        next_logits = self.target(next_states).detach()
        targets = rewards + self.gamma * (1 - dones) * next_logits.max(dim=1).values

        pred = logits.gather(1, actions.unsqueeze(1)).squeeze(1)
        td_error = (pred - targets).pow(2)
        loss = (weights * td_error).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 5.0)
        self.optimizer.step()

        self.replay_buffer.update_priorities(indices, td_error.detach().cpu().numpy())

    def update_target(self, tau=0.01):
        for target_param, param in zip(self.target.parameters(), self.policy.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
