import torch
from typing import Dict, Any
from experiments.logger import setup_logger

REWARDS = {
    "crash": 5,
    "new_state": 10,
    "repeated_state": -5,
    "text_input_penalty": -2,
    "app_left": -10,
    "invalid_state": -1,
    "default_penalty": -1
}
class RewardAnalyzer:
    def __init__(self, log_dir: str = "app/logs/", emulator_name: str = "Unknown", app_name: str = "Unknown"):
        self.logger = setup_logger(f"{log_dir}/reward_analyzer.log", emulator_name=emulator_name, app_name=app_name)
        self.seen_states = set()
        self.seen_crashes = set()
        self.last_action = None
        self.last_widget = None
        self.consecutive_text_inputs = 0

    def calculate_reward(self, state, next_state, action, crash_occurred, crash_id, app_left):
        reward = 0

        # Handle app crashes or exits
        if app_left:
            reward = -10
        elif crash_occurred:
            if crash_id not in self.seen_crashes:
                reward = 5
                self.seen_crashes.add(crash_id)
            else:
                reward = -1
        elif next_state is not None:
            state_str = ''.join(map(str, next_state.int().tolist()))
            if state_str not in self.seen_states:
                self.seen_states.add(state_str)
                reward = 10
            else:
                reward = -5

            # Penalize excessive repeated text input
            if action['type'] == 'text_input':
                current_widget = action.get('widget_index')
                if self.last_action == 'text_input' and self.last_widget == current_widget:
                    self.consecutive_text_inputs += 1
                    reward -= 2 * self.consecutive_text_inputs
                else:
                    self.consecutive_text_inputs = 1
            else:
                self.consecutive_text_inputs = 0

            self.last_action = action['type']
            self.last_widget = action.get('widget_index')
        else:
            reward = -1  # default penalty for invalid or stuck state

        self.logger.info(f"Reward calculated: {reward}")
        return reward

    # def calculate_reward(self, state, next_state, action, crash_occurred, crash_id, app_left):
    #     reward = 0

    #     # Handle app crashes or exits
    #     if app_left:
    #         reward = -100
    #     elif crash_occurred:
    #         reward = 100 if crash_id not in self.seen_crashes else -10
    #         self.seen_crashes.add(crash_id)
    #     elif next_state is not None:
    #         # Reward new states
    #         state_str = ''.join(map(str, next_state.int().tolist()))
    #         if state_str not in self.seen_states:
    #             self.seen_states.add(state_str)
    #             reward = 1000
    #         else:
    #             reward = -100

    #         # Penalize consecutive text inputs on the same widget
    #         if action['type'] == 'text_input':
    #             current_widget = action.get('widget_index')
    #             if (self.last_action == 'text_input' and 
    #                 self.last_widget == current_widget):
    #                 self.consecutive_text_inputs += 1
    #                 if self.consecutive_text_inputs > 1:
    #                     reward -= 50 * self.consecutive_text_inputs
    #             else:
    #                 self.consecutive_text_inputs = 1
    #         else:
    #             self.consecutive_text_inputs = 0

    #         self.last_action = action['type']
    #         self.last_widget = action.get('widget_index')
    #     else:
    #         reward = -10


    #     self.logger.info(f"Reward calculated: {reward}")
    #     return reward