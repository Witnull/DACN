import os
import sys
import time
import subprocess
import pathlib as pathlib
from typing import Dict, Any
from selenium.common.exceptions import WebDriverException
from experiments.logger import setup_logger
from experiments.action_extractor import ActionExtractor
from experiments.gui_embedder import GUIEmbedder
from experiments.reward_analyzer import RewardAnalyzer
from experiments.test2 import DQNAgent, train_macro_generator
from experiments.utils.apk_analyzer import apk_analyzer
from experiments.utils.emulator_n_appium_controller import EmulatorController, AppiumManager
from lxml import etree as ET
from experiments.utils.path_config import ADB_PATH, EMULATOR_PATH, ANDROID_HOME, JAVA_HOME, NODE_PATH, APKSIGNER_PATH

import datetime
log_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


class EnviromentHandler:
    def __init__(self, emulator_name: str, emulator_port: str, appium_port: int, apk_path: str):
        self.emulator_name = emulator_name
        self.appium_port = appium_port
        self.emulator_port = emulator_port
        self.apk_path = apk_path
        self.app_name = pathlib.Path(apk_path).name.split('.')[0]
        self.coverage_dict_template = {}

        self.model_dir = "app/model/"
        os.makedirs(self.model_dir, exist_ok=True)
        self.ver ="t"+str(log_time)
        self.model_path = f"{self.model_dir}/model_{self.ver}.pth"
        self.replay_buffer_path = f"{self.model_dir}/replay_buffer_{self.ver}.pkl"
        # Logging setup
        self.master_log_dir = f"app/logs/{self.emulator_name}_{self.app_name}_t{log_time}"
        os.makedirs(self.master_log_dir, exist_ok=True)
        self.logger = setup_logger(f"{self.master_log_dir}/emulator_handler.log", emulator_name=self.emulator_name, app_name=self.app_name)
        self.logger.info(f"EnviromentHandler initialized for {self.emulator_name} and {self.app_name}")

        self.appium_manager = AppiumManager(appium_port=self.appium_port, apk_path=self.apk_path, coverage_dict_template=self.coverage_dict_template, log_dir = self.master_log_dir, emulator_name=self.emulator_name, app_name=self.app_name)
        self.emulator_controller = EmulatorController(emulator_name=self.emulator_name, emulator_port=self.emulator_port, log_dir = self.master_log_dir, app_name=self.app_name)

        self.max_activities = 50
        self.max_widgets = 200

        self.state_dim = self.max_activities + self.max_widgets + 4 + 384  # +4 for orientation, network, focused_text, scrollable
        self.action_vector_dim = 15  # Fixed action vector size

        self.gui_embedder = GUIEmbedder(log_dir=self.master_log_dir, emulator_name=self.emulator_name, app_name=self.app_name, max_activities=self.max_activities, max_widgets=self.max_widgets)
        self.dqn_agent = DQNAgent(state_dim=self.state_dim, action_vector_dim=self.action_vector_dim, log_dir=self.master_log_dir, emulator_name=self.emulator_name, app_name=self.app_name)

        self.action_extractor = None  # Initialized after driver setup
        self.reward_analyzer = RewardAnalyzer(log_dir=self.master_log_dir, emulator_name=self.emulator_name, app_name=self.app_name)
        self.device_name = None # init after emulator start

        # Environment paths
        self.android_home = ANDROID_HOME
        self.emulator_path = EMULATOR_PATH
        self.adb_path = ADB_PATH
        self.apksigner_path = APKSIGNER_PATH
        self.java_home = JAVA_HOME
        self.node_path = NODE_PATH

    def analyze_apk(self) ->bool:
        try:
            self.exported_activities,  self.services,  self.receivers,  self.providers,  self.string_activities,  self.app_package = apk_analyzer(self.apk_path, self.coverage_dict_template)
            self.logger.info(f"APK analysis completed: \n{self.exported_activities}\n {self.services}\n {self.receivers}\n {self.providers}\n {self.string_activities}\n {self.app_package}")
            return True
        except Exception as e:
            self.logger.error(f"APK analysis failed: {str(e)}")
            return False
        
    def clear_logcat(self):
        """Clear the logcat to prepare for new logs."""
        subprocess.run([self.adb_path, "logcat", "-c"], check=True)

    def get_logcat(self):
        """Retrieve the current logcat output."""
        output = subprocess.check_output([self.adb_path, "logcat", "-d"])
        return output.decode()

    def parse_logcat_for_crash(self, logcat: str):
        """Parse logcat output for signs of a crash from the target app."""
        lines = logcat.splitlines()
        relevant_lines = [line for line in lines if self.app_package in line]

        for i, line in enumerate(relevant_lines):
            if "FATAL EXCEPTION" in line or "ANR in" in line:
                crash_block = relevant_lines[i:i + 10]
                crash_id = "\n".join(crash_block)
                self.logger.error(f"Crash detected: {crash_id} \n#######\n\n {crash_block} \n\n#########")
                return True, crash_id
        return False, None

    
    def check_app_status(self):
        """Check if the testing app is still in focus."""
        try:
            current_package = self.appium_manager.driver.current_package
            return current_package == self.app_package
        except WebDriverException:
            return False
        
    def restart_app(self):
        try:
            self.logger.info(f"Restarting app: {self.app_package}")
            # Terminate the app if it's running
            self.appium_manager.driver.terminate_app(self.app_package)
            time.sleep(1)
            # Start the app's main activity
            self.appium_manager.driver.activate_app(self.app_package)
            time.sleep(2)  # Let the app stabilize
            self.logger.info("App restarted successfully")
        except Exception as e:
            self.logger.error(f"Failed to restart app: {str(e)}")
            raise


    def start_emulator(self):
        if self.emulator_controller.start_emulator():
            self.device_name = self.emulator_controller.get_device_name()
            self.logger.info(f"Emulator started: {self.device_name}")
            self.emulator_controller.install_appium_apks()
        else:
            self.logger.error("Failed to start emulator")
            sys.exit(1)

    def start_appium(self):
        if not self.analyze_apk():
            self.logger.error("APK analysis failed, cannot start Appium")
            sys.exit(1)
        if self.appium_manager.start_appium_server():
            if self.appium_manager.connect(self.device_name, self.string_activities,  self.app_package):
                self.action_extractor = ActionExtractor(self.appium_manager.driver, log_dir=self.master_log_dir, emulator_name=self.emulator_name, app_name=self.app_name)
                self.logger.info("Appium server started and connected successfully")
            else:
                self.logger.error("Failed to connect to Appium server")
                sys.exit(1)
        else:
            self.logger.error("Failed to start Appium server")
            sys.exit(1)

   
    def check_emulator_status(self) -> bool:
        return self.emulator_controller.check_emulator_status(self.device_name)
    
    def cleanup_emulator(self):
        self.emulator_controller.cleanup_emulator(self.device_name)
        self.appium_manager.cleanup_appium()
        self.action_extractor = None  # Reset action extractor

    def perform_action(self, action: Dict[str, Any]):
        try:
            if action['type'] == 'touch':
                widget_index = action['widget_index']
                gui_hierarchy = self.appium_manager.driver.page_source
                root = ET.fromstring(gui_hierarchy.encode('utf-8'))
                for element in root.findall(".//*[@clickable='true']"):
                    widget_key = (element.get('class'), element.get('text', ''))
                    if self.gui_embedder.widget_dict.get(widget_key) == widget_index:
                        xpath = self.action_extractor._generate_xpath(element)
                        self.appium_manager.driver.find_element(by='xpath', value=xpath).click()
                        break
            elif action['type'] == 'gesture':
                widget_index = action['widget_index']
                gesture_type = action['parameters']['gesture_type']
                direction = action['parameters']['direction']
                if gesture_type == 'swipe':
                    gui_hierarchy = self.appium_manager.driver.page_source
                    root = ET.fromstring(gui_hierarchy.encode('utf-8'))
                    for element in root.findall(".//*[@scrollable='true']"):
                        widget_key = (element.get('class'), element.get('text', ''))
                        if self.gui_embedder.widget_dict.get(widget_key) == widget_index:
                            bounds = element.get('bounds', '[0,0][0,0]')
                            x, y = self.action_extractor._parse_bounds(bounds)
                            end_x = x + 500 if direction == 'right' else x - 500 if direction == 'left' else x
                            end_y = y + 500 if direction == 'down' else y - 500 if direction == 'up' else y
                            self.appium_manager.driver.swipe(start_x=x, start_y=y, end_x=end_x, end_y=end_y, duration=100)
                            break
            elif action['type'] == 'text_input':
                widget_index = action['widget_index']
                text = action['parameters']['text']
                gui_hierarchy = self.appium_manager.driver.page_source
                root = ET.fromstring(gui_hierarchy.encode('utf-8'))
                for element in root.findall(".//*[@class='android.widget.EditText']"):
                    widget_key = (element.get('class'), element.get('text', ''))
                    if self.gui_embedder.widget_dict.get(widget_key) == widget_index:
                        xpath = self.action_extractor._generate_xpath(element)
                        self.appium_manager.driver.find_element(by='xpath', value=xpath).send_keys(text)
                        break
            elif action['type'] == 'system':
                system_action = action['action']
                if system_action == 'toggle_network':
                    subprocess.run(["adb", "shell", "svc", "wifi", "enable" if self.gui_embedder.get_network_status() == 0 else "disable"])
                elif system_action == 'rotate_screen':
                    subprocess.run(["adb", "shell", "settings", "put", "system", "user_rotation", "1" if self.gui_embedder.get_device_orientation() == 0 else "0"])
            self.logger.debug(f"Performed action: {action['type']}")
        except Exception as e:
            self.logger.error(f"Failed to perform action {action}: {str(e)}")

    def run_testing(self, max_steps: int = 100, episodes: int = 10):
        """Run the RL-based GUI testing loop with reward calculation and graph generation."""
        if os.path.exists(self.model_path):
            self.dqn_agent.load_model(self.model_path)
        if os.path.exists(self.replay_buffer_path):
            self.dqn_agent.load_replay_buffer(self.replay_buffer_path)

        self.start_emulator()
        self.start_appium()
        try:
            for episode in range(episodes):
                self.logger.info(f"Starting episode {episode}")
                for step in range(max_steps):
                    if not self.check_emulator_status():
                        self.logger.error("Emulator offline, attempting to restart")
                        self.cleanup_emulator()
                        self.start_emulator()
                        self.start_appium()

                    time.sleep(.5)  # Wait for GUI stability
                    gui_hierarchy = self.appium_manager.driver.page_source
                    state = self.gui_embedder.embed(gui_hierarchy)

                    #self.logger.debug(f"widget_dict before extract_actions: {self.gui_embedder.widget_dict}")
                    actions, action_vectors = self.action_extractor.extract_actions(gui_hierarchy, self.gui_embedder.widget_dict)
                    
                    if not actions:
                        self.logger.info("No actions available, resetting app")
                        self.restart_app()
                        reward = -10.0
                        self.dqn_agent.store(state, -1, reward, state, True, None)
                        continue

                    action_idx = self.dqn_agent.act(state, actions, action_vectors)
                    action = actions[action_idx]

                    self.clear_logcat()
                    self.perform_action(action)
                    time.sleep(.5)
                    logcat = self.get_logcat()
                    crash_occurred, crash_id = self.parse_logcat_for_crash(logcat)
                    app_left = not self.check_app_status()

                    if app_left and not crash_occurred:
                        self.restart_app()
                        next_state = self.gui_embedder.embed(self.appium_manager.driver.page_source)
                        next_gui_hierarchy = self.appium_manager.driver.page_source
                    elif not crash_occurred:
                        next_gui_hierarchy = self.appium_manager.driver.page_source
                        next_state = self.gui_embedder.embed(next_gui_hierarchy)
                    else:

                        self.logger.error("Crash occurred, restarting app")
                        self.logger.error(f"Action do crash: {action['type']}")
                        self.logger.error(f"Location: {action['widget_index']}, {action['parameters']}")
                        self.logger.error(f"Crash ID: {crash_id}")
                        self.logger.error(f"Logcat: {logcat}")

                        self.restart_app()
                        next_state = None
                        next_gui_hierarchy = None

                    # Extract next action vectors
                    next_action_vectors = (
                        self.action_extractor.extract_actions(next_gui_hierarchy, self.gui_embedder.widget_dict)[1]
                        if next_gui_hierarchy else []
                    )
                    reward = self.reward_analyzer.calculate_reward(state, next_state, action, crash_occurred, crash_id, app_left)
                    if next_state is not None and len(next_action_vectors) > action_idx:
                        intrinsic_loss = self.dqn_agent.icm(state, next_action_vectors[action_idx], next_state)
                        intrinsic = intrinsic_loss.item()
                        reward += self.dqn_agent.beta * intrinsic
                    done = step == max_steps - 1
                    
                    # self.logger.debug(f"State: {state}, Action: {action}, Reward: {reward}, Next State: {next_state}, Done: {done}")
                    # self.logger.debug(f"Action vectors: {action_vectors}")

                    self.dqn_agent.store(state, action_idx, reward, next_state, done, next_action_vectors)

                    self.dqn_agent.train()
                    if len(self.dqn_agent.memory) % 8 ==1 and len(self.dqn_agent.memory) > 100:
                        mmodel = train_macro_generator(self.dqn_agent.memory, self.state_dim, self.action_vector_dim)
                        self.dqn_agent.macro_generator.load_state_dict(mmodel.state_dict())
                    
                    self.logger.warning(f"Episode {episode}, Step {step}: Action {action['type']}")
                    self.logger.warning(f"Reward: {reward}, Crash: {crash_occurred}, App left: {app_left}")
                   

                # Save graph and generate HTML visualization at the end of each episode
                #self.dqn_agent.stgraph.save_graph(episode)
                self.dqn_agent.save_model(self.model_path)
                self.dqn_agent.save_replay_buffer(self.replay_buffer_path)
                self.dqn_agent.hidden_state = None
                
                # self.logger.warning(f"Code cov: {self.check_coverage()}") 

          
        finally:
            self.cleanup_emulator()
            self.logger.warning("Testing completed")