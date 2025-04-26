import torch
import math
import hashlib
from typing import Dict, List, Any, Tuple
from collections import defaultdict
from experiments.logger import setup_logger
from experiments.graph_generator import GraphGenerator
import networkx as nx

class StateTransitionGraph:
    def __init__(self, log_dir: str = "app/logs/", emulator_name: str = "Unknown", app_name: str = "Unknown"):
        self.logger = setup_logger(f"{log_dir}/state_transition_graph.log", emulator_name=emulator_name, app_name=app_name)
        self.graph_generator = GraphGenerator(log_dir, emulator_name, app_name)
        self.state_hashes = {}  # Map state tensors to their hash IDs
        self.logger.info("StateTransitionGraph initialized")
        self.graph = nx.DiGraph()
        self.motifs = []  # stored macro patterns

    def _get_state_id(self, state: torch.Tensor) -> str:
        """Generate a unique ID for a state tensor using its hash."""
        state_bytes = state.cpu().numpy().tobytes()
        state_hash = hashlib.md5(state_bytes).hexdigest() 
        self.state_hashes[state_hash] = state
        return state_hash

    # def add_transition(self, state: torch.Tensor, action_idx: int, next_state: torch.Tensor, action: Dict[str, Any] = None):
    #     """Add a transition to the graph."""
    #     try:
    #         state_id = self._get_state_id(state)
    #         next_state_id = self._get_state_id(next_state) if next_state is not None else "terminal"
    #         # Use action if provided, else create a placeholder
    #         action_dict = action if action is not None else {'type': 'unknown', 'action_idx': action_idx}
    #         self.graph_generator.add_transition(state_id, action_dict, next_state_id)
    #         self.logger.debug(f"Added transition: {state_id} -> {next_state_id} with action {action_dict['type']}")
    #     except Exception as e:
    #         self.logger.error(f"Failed to add transition: {str(e)}")
    def add_transition(self, state_hash: str, action_str: str, next_state_hash: str):
        self.graph.add_edge(state_hash, next_state_hash, action=action_str)

    def find_frequent_subgraphs(self, min_support=3):
        paths = []
        for start in self.graph.nodes:
            for path in nx.single_source_shortest_path(self.graph, start, cutoff=3).values():
                if len(path) > 2:
                    paths.append(tuple(path))
        counter = defaultdict(int)
        for p in paths:
            counter[p] += 1
        self.motifs = [p for p, c in counter.items() if c >= min_support]

    def match_motif(self, current_state_hash: str):
        for motif in self.motifs:
            if current_state_hash in motif:
                idx = motif.index(current_state_hash)
                if idx + 1 < len(motif):
                    next_node = motif[idx + 1]
                    action = self.graph[current_state_hash][next_node]['action']
                    return action  # return action string or lookup actual action
        return None

    def save_graph(self, episode: int):
        """Save the graph in GraphML format and generate HTML visualization."""
        try:
            graphml_file = f"state_graph_episode_{episode}.graphml"
            html_file = f"state_graph_episode_{episode}.html"
            self.graph_generator.save_graph(graphml_file)
            self.graph_generator.generate_html(html_file)
            self.logger.info(f"Saved graph and visualization for episode {episode}")
        except Exception as e:
            self.logger.error(f"Failed to save graph: {str(e)}")

class MCTS:
    def __init__(self, dqn_agent, graph: StateTransitionGraph, num_simulations: int = 100,
                 c1: float = 1.0, c2: float = 1.0, log_dir: str = "app/logs/",
                 emulator_name: str = "Unknown", app_name: str = "Unknown"):
        self.dqn_agent = dqn_agent
        self.graph = graph
        self.num_simulations = num_simulations
        self.c1 = c1
        self.c2 = c2
        self.logger = setup_logger(
            f"{log_dir}/mcts.log", emulator_name=emulator_name, app_name=app_name
        )
        self.logger.info(f"MCTS initialized with {num_simulations} simulations, c1={c1}, c2={c2}")

    def add_transition(self, state, action_idx: int, next_state, reward: float, action: Dict[str, Any] = None):
        """Add a transition to the state graph for MCTS planning."""
        try:
            self.state_graph.add_transition(state, action_idx, next_state, action)
            self.logger.debug(f"MCTS: Added transition for action {action_idx}")
        except Exception as e:
            self.logger.error(f"Failed to add transition: {str(e)}")

    def ucb_score(self, q_value: float, prior: float, parent_visits: int, visits: int) -> float:
        exploration_term = self.c1 + math.log((parent_visits + self.c2 + 1) / self.c2)
        ucb = q_value + prior * (math.sqrt(parent_visits) / (1 + visits)) * exploration_term
        return ucb

    def search(self, state: torch.Tensor, actions: List[Dict[str, Any]]) -> int:
        q_values = self.dqn_agent.get_q_values(state, actions)
        priors = torch.softmax(q_values, dim=0).tolist()

        visits = defaultdict(int)
        total_rewards = defaultdict(float)

        for _ in range(self.num_simulations):
            current_state = state
            selected_action_idx = None
            path = []  # Store (state, action_idx) pairs

            while True:
                transitions = self.graph.get_transitions(current_state)
                available_actions = [
                    idx for idx in range(len(actions))
                    if idx in transitions or transitions[idx][0] is None
                ]
                if not available_actions:
                    break

                parent_visits = sum(transitions[idx][1] for idx in available_actions)
                ucb_scores = {}
                for idx in available_actions:
                    next_state_key, action_visits, total_reward = transitions[idx]
                    q_value = total_reward / action_visits if action_visits > 0 else 0
                    prior = priors[idx]
                    ucb_scores[idx] = self.ucb_score(q_value, prior, parent_visits, action_visits)

                selected_action_idx = max(ucb_scores, key=ucb_scores.get)
                path.append((current_state, selected_action_idx))
                next_state_key, _, _ = transitions[selected_action_idx]
                if next_state_key is None:
                    break
                # Since we don't have the actual next_state tensor, simulate using Q-value
                current_state = None  # Cannot proceed without next_state tensor
                break

            simulated_reward = q_values[selected_action_idx].item() if selected_action_idx is not None else 0

            for state_tensor, action_idx in path:
                state_key = self.graph.state_to_key(state_tensor)
                visits[(state_key, action_idx)] += 1
                total_rewards[(state_key, action_idx)] += simulated_reward

        action_visits = defaultdict(int)
        state_key = self.graph.state_to_key(state)
        for (s_key, action_idx), visit_count in visits.items():
            if s_key == state_key:
                action_visits[action_idx] += visit_count

        if not action_visits:
            self.logger.warning("No actions simulated, selecting randomly")
            return torch.randint(0, len(actions), (1,)).item()

        best_action_idx = max(action_visits, key=action_visits.get)
        self.logger.info(f"Selected action {best_action_idx} with {action_visits[best_action_idx]} visits")
        return best_action_idx