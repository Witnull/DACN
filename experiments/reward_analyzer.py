import torch
from typing import Dict, Any
from experiments.logger import setup_logger
from lxml import etree as ET
import logging

REWARDS = {
    "crash": 20,
    "new_state": 10,
    "repeated_state": -5,
    "text_input_penalty": -2,
    "app_left": -10,
    "invalid_state": -1,
    "default_penalty": -1,
    "debuggable_app_found": 5, # Penalty for finding a debuggable app
    "cleartext_traffic_found": 10, # Penalty for finding cleartext traffic
    "dangerous_permission_found": 10, # Penalty for finding dangerous permissions
}
class RewardAnalyzer:
    def __init__(self, log_dir: str = "app/logs/", emulator_name: str = "Unknown", app_name: str = "Unknown", gui_embedder=None, debuggable: bool = False, uses_cleartext_traffic: bool = False, permissions: list = [], logger=None):
        if logger is None:
            self.logger = setup_logger(f"{log_dir}/reward_analyzer.log", emulator_name=emulator_name, app_name=app_name)
        else:
            self.logger = logger
        self.seen_states = set()
        self.seen_widgets = set()
        self.seen_crashes = set()
        self.last_action = None
        self.last_widget = None
        self.consecutive_text_inputs = 0
        self.gui_embedder = gui_embedder
        self.debuggable = debuggable
        self.uses_cleartext_traffic = uses_cleartext_traffic
        self.permissions = permissions

    def calculate_reward(self, state, next_state, action, crash_occurred, crash_logs, app_left, next_gui_hierarchy=None):
        reward = 0

        # Handle app crashes or exits
        if app_left:
            reward = REWARDS["app_left"]
        elif crash_occurred:
            pass
            # if crash_id not in self.seen_crashes:
            #     reward = REWARDS["crash"]
            #     self.seen_crashes.add(crash_id)
            # else:
            #     reward = REWARDS["repeated_state"]
        elif next_state is not None:
            state_str = ".".join(map(str, next_state.int().tolist()))
            if state_str not in self.seen_states:
                self.seen_states.add(state_str)
                reward = REWARDS["new_state"]
            else:
                reward = REWARDS["repeated_state"]

            # Reward for new GUI elements discovered
            if next_gui_hierarchy and self.gui_embedder:
                root = ET.fromstring(next_gui_hierarchy.encode("utf-8"))
                clickable_elements = (
                    root.findall(".//*[@clickable=\'true\']") +
                    root.findall(".//*[@scrollable=\'true\']") +
                    root.findall(".//*[@class=\'android.widget.EditText\']")
                )
                for element in clickable_elements:
                    widget_type = element.get("class", ".")
                    if "EditText" in widget_type:
                        widget_id = element.get("resource-id", ".")
                        widget_key = (widget_type, widget_id)
                    else:
                        widget_text = element.get("text", ".")
                        widget_key = (widget_type, widget_text)
                    
                    if widget_key not in self.seen_widgets:
                        self.seen_widgets.add(widget_key)
                        reward += 5  # Additional reward for new widget

            # Penalize excessive repeated text input
            if action["type"] == "text_input":
                current_widget = action.get("widget_index")
                if self.last_action == "text_input" and self.last_widget == current_widget:
                    self.consecutive_text_inputs += 1
                    reward += REWARDS["text_input_penalty"] * self.consecutive_text_inputs
                else:
                    self.consecutive_text_inputs = 1
            else:
                self.consecutive_text_inputs = 0

            self.last_action = action["type"]
            self.last_widget = action.get("widget_index")
        else:
            reward = REWARDS["invalid_state"]  # default penalty for invalid or stuck state

        # Incorporate APK analysis findings into reward
        if self.debuggable:
            reward += REWARDS["debuggable_app_found"]
            self.logger.warning("Debuggable app found. Applying penalty.")
        if self.uses_cleartext_traffic:
            reward += REWARDS["cleartext_traffic_found"]
            self.logger.warning("App uses cleartext traffic. Applying penalty.")
        
        # Define a list of dangerous permissions (example, can be expanded)
        dangerous_permissions = [
            "android.permission.READ_CONTACTS",
            "android.permission.WRITE_CONTACTS",
            "android.permission.GET_ACCOUNTS",
            "android.permission.READ_CALENDAR",
            "android.permission.WRITE_CALENDAR",
            "android.permission.ACCESS_FINE_LOCATION",
            "android.permission.ACCESS_COARSE_LOCATION",
            "android.permission.RECORD_AUDIO",
            "android.permission.READ_PHONE_STATE",
            "android.permission.CALL_PHONE",
            "android.permission.READ_CALL_LOG",
            "android.permission.WRITE_CALL_LOG",
            "android.permission.ADD_VOICEMAIL",
            "android.permission.USE_SIP",
            "android.permission.PROCESS_OUTGOING_CALLS",
            "android.permission.BODY_SENSORS",
            "android.permission.SEND_SMS",
            "android.permission.RECEIVE_SMS",
            "android.permission.READ_SMS",
            "android.permission.RECEIVE_WAP_PUSH",
            "android.permission.RECEIVE_MMS",
            "android.permission.READ_EXTERNAL_STORAGE",
            "android.permission.WRITE_EXTERNAL_STORAGE"
        ]

        for perm in self.permissions:
            if perm in dangerous_permissions:
                reward += REWARDS["dangerous_permission_found"]
                self.logger.warning(f"Dangerous permission found: {perm}. Applying penalty.")

        self.logger.info(f"Reward calculated: {reward}")
        return reward

