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
from experiments.utils.emulator_n_appium_controller import (
    EmulatorController,
    AppiumManager,
)
from experiments.utils.path_config import (
    ADB_PATH,
)
from experiments.logcat_extractor import LogcatExtractor
from lxml import etree as ET
from acvtool.acvtool import get_parser, run_actions

# command_map = {
#     "instrument": acv.instrument,
#     "install": acv.install,
#     "uninstall": acv.uninstall,
#     "activate": acv.activate,
#     "start": acv.start,
#     "stop": acv.stop,
#     "snap": acv.snap,
#     "flush": acv.flush,
#     "calculate": acv.calculate,
#     "pull": acv.pull,
#     "cover-pickles": acv.cover_pickles,
#     "report": acv.report,
#     "sign": acv.sign,
#     "build": acv.build,
#     "shrink": acv.shrink,
#     "smali": acv.smali,
# }
import datetime


log_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


class EnviromentHandler:
    def __init__(
        self, emulator_name: str, emulator_port: str, appium_port: int, apk_path: str
    ):
        self.emulator_name = emulator_name
        self.appium_port = appium_port
        self.emulator_port = emulator_port
        self.apk_path = apk_path
        self.app_name = pathlib.Path(apk_path).name.split(".")[0]
        self.acv_intructed_apk_path = ""  # path to the instrumented apk, will define after ACVTool instrumentation
        # Environment paths
        self.adb_path = ADB_PATH

        # APK analysis results
        self.debuggable = False
        self.uses_cleartext_traffic = False
        self.permissions = []
        # Logging setup
        self.ver = "t" + str(log_time)

        # Create directories for logs and models
        self.master_log_dir = f"Logs/{self.emulator_name}_{self.app_name}_{self.ver}"
        os.makedirs(self.master_log_dir, exist_ok=True)
        self.model_dir = f"{self.master_log_dir}/model"
        os.makedirs(self.model_dir, exist_ok=True)
        self.inst_apk_dir = "apk/instr"
        os.makedirs(self.inst_apk_dir, exist_ok=True)
        self.acv_workdir = "apk/instr/acv_temp"  # will set after apk analyze

        # Set saving paths
        self.model_path = f"{self.model_dir}/model_{self.ver}.pth"
        self.replay_buffer_path = f"{self.model_dir}/replay_buffer_{self.ver}.pkl"

        self.logger = setup_logger(
            f"{self.master_log_dir}/emulator_handler.log",
            emulator_name=self.emulator_name,
            app_name=self.app_name,
        )

        self.logger.info(
            f"EnviromentHandler initialized for {self.emulator_name} and {self.app_name}"
        )

        # Initialize Appium and Emulator controllers
        self.appium_manager = None  # Initialized after APK ACVTool instrumentation
        self.emulator_controller = EmulatorController(
            emulator_name=self.emulator_name,
            emulator_port=self.emulator_port,
            log_dir=self.master_log_dir,
            app_name=self.app_name,
        )

        ##############################################
        # Initialize/set model variables
        #
        ##############################################

        self.max_activities = 50  # Maximum number of activities to track
        self.max_widgets = 300  # Maximum number of widgets to track
        self.max_orientation = 4  # up down left right
        # State dimemsion calculation
        self.state_dim = (
            self.max_activities + self.max_widgets + 4 + 384
        )  # +4 for orientation, network, focused_text, scrollable

        self.action_vector_dim = 15  # Fixed action vector size

        # Initialize components
        self.gui_embedder = GUIEmbedder(
            log_dir=self.master_log_dir,
            emulator_name=self.emulator_name,
            app_name=self.app_name,
            max_activities=self.max_activities,
            max_widgets=self.max_widgets,
        )

        self.dqn_agent = DQNAgent(
            state_dim=self.state_dim,
            action_vector_dim=self.action_vector_dim,
            log_dir=self.master_log_dir,
            emulator_name=self.emulator_name,
            app_name=self.app_name,
        )

        self.action_extractor = None  # Initialized after driver setup

        self.reward_analyzer = RewardAnalyzer(
            log_dir=self.master_log_dir,
            emulator_name=self.emulator_name,
            app_name=self.app_name,
            gui_embedder=self.gui_embedder,
            debuggable=self.debuggable,
            uses_cleartext_traffic=self.uses_cleartext_traffic,
            permissions=self.permissions,
        )

        self.device_name = None  # Will be set after emulator starts
        self.logcat_extractor = None  # Will be initialized after emulator starts

        self.acv_parser = get_parser()

        ###################################### EO__init__

    def analyze_apk(self) -> bool:
        ######################################
        #   ANALYZE APK
        #   This step analyzes the APK to extract activities, services, receivers, providers, and permissions.
        #   Before running the tests
        ######################################
        try:
            (
                self.exported_activities,
                self.services,
                self.receivers,
                self.providers,
                self.app_package,
            ) = apk_analyzer(self.apk_path)

            self.logger.info(
                f"APK analysis completed: \nExported Activities: {self.exported_activities}\n Services: {self.services}\n Receivers: {self.receivers}\n Providers: {self.providers}\n String Activities: {self.string_activities}\n App Package: {self.app_package}\n Debuggable: {self.debuggable}\n Uses Cleartext Traffic: {self.uses_cleartext_traffic}\n Permissions: {self.permissions}"
            )
            self.acv_workdir = f"{self.inst_apk_dir}/{self.app_package}"
            return True
        except Exception as e:
            self.logger.error(f"APK analysis failed: {str(e)}")
            return False

    def _intr_apk_w_acvtool(self):
        try:
            if os.path.exists(
                f"{self.inst_apk_dir}/{self.app_package}/instr_{self.app_package}.apk"
            ):
                self.logger.info(
                    f"Instrumented APK already exists at {self.inst_apk_dir}. Skipping instrumentation."
                )
                self.acv_intructed_apk_path = f"{self.inst_apk_dir}/{self.app_package}/instr_{self.app_package}.apk"

                return True
            # Create a proper Namespace object for ACVTool
            instrument_args_list = [
                "instrument",
                self.apk_path,
                "--wd",
                self.acv_workdir,
                "--force",
                "-g",
                "instruction",
            ]

            self.logger.info(f"Running ACVTool with args: {instrument_args_list}")
            instrument_args = self.acv_parser.parse_args(instrument_args_list)

            # Run the action with the properly parsed arguments
            run_actions(self.acv_parser, args=instrument_args)

            self.acv_intructed_apk_path = os.path.join(
                self.acv_workdir, f"instr_{self.app_package}.apk"
            )
            if os.path.exists(self.acv_intructed_apk_path):
                self.logger.info(
                    f"APK instrumented with ACVTool -> {self.acv_intructed_apk_path}"
                )
                return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to instrument APK with ACVTool: {str(e)}")
            raise Exception(
                "APK instrumentation failed. Please check the APK path and ACVTool setup."
            )
        # 2 Install the instrumented APK in the Android emulator or device. [install ]
        # 3 Activate the app for coverage measurement [activate <package_name>] (alternatively, [start <package_name>])
        # 4 Test the application (launch it!)

    def check_app_status(self) -> bool:
        """Check if the testing app is still in focus. Open"""
        try:
            current_package = self.appium_manager.driver.current_package
            return current_package == self.app_package
        except WebDriverException as we:
            self.logger.error(f"WebDriverException: {str(we)}")
            return False
        except Exception as e:
            self.logger.error(f"Failed to check app status: {str(e)}")
            return False

    def terminate_app(self) -> bool:
        """Stop the app on the emulator."""
        for i in range(3):
            try:
                self.logger.info(f"Attempt {i} stopping app: {self.app_package}")
                # Terminate the app if it's running
                self.appium_manager.driver.terminate_app(self.app_package)
                time.sleep(2)  # Let the app close properly
                self.logger.info("App stopped successfully")
                return True
            except Exception as e:
                self.logger.error(f"Failed to stop app: {str(e)}")
        raise Exception(
            "Failed to stop app after multiple attempts. Please check the app setup."
        )

    def start_app(self) -> bool:
        """Start the app on the emulator."""
        for i in range(3):
            try:
                self.logger.info(f"Attempt {i} starting app: {self.app_package}")
                # Start the app's main activity
                self.appium_manager.driver.activate_app(self.app_package)
                time.sleep(2)  # Let the app stabilize
                self.logger.info("App started successfully")
                return True
            except Exception as e:
                self.logger.error(f"Failed to start app: {str(e)}")
        raise Exception(
            "Failed to start app after multiple attempts. Please check the app setup."
        )

    def restart_app(self) -> bool:
        try:
            self.logger.info(f"Restarting app: {self.app_package}")
            # Terminate the app if it's running
            self.terminate_app()
            # Start the app's main activity
            self.start_app()
            self.logger.info("App restarted successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to restart app: {str(e)}")
            return False

    def start_emulator(self) -> bool:
        for i in range(3):
            try:
                if self.check_emulator_status():
                    self.logger.warning(
                        f"Emulator {self.emulator_name} is already running. Cleaning up..."
                    )
                    self.cleanup_emulator()

                self.logger.info(f"Attempt {i + 1} to start emulator")
                if self.emulator_controller.start_emulator():
                    self.device_name = self.emulator_controller.get_device_name()
                    self.logger.info(f"Emulator started: {self.device_name}")

                    # Initialize LogcatExtractor only after we have a valid device name
                    self.logcat_extractor = LogcatExtractor(
                        adb_path=self.adb_path,
                        device_udid=self.device_name,
                        app_name=self.app_name,
                        logdir=self.master_log_dir,
                    )

                    self.emulator_controller.install_appium_apks()  # have auto retry
                    return True
            except Exception as e:
                self.logger.error(f"Failed to start emulator: {str(e)}")
            time.sleep(10)
        raise Exception(
            "Failed to start emulator after multiple attempts. Please check the emulator setup."
        )

    def start_appium(self) -> bool:
        if not self.analyze_apk():
            self.logger.error("APK analysis failed, cannot start Appium")
            raise Exception("APK analysis failed, cannot start Appium")
        if not self._intr_apk_w_acvtool():
            self.logger.error("APK instrumentation failed, cannot start test")
            raise Exception("APK instrumentation failed, cannot start test")

        self.appium_manager = AppiumManager(
            appium_port=self.appium_port,
            apk_path=self.acv_intructed_apk_path,
            log_dir=self.master_log_dir,
            emulator_name=self.emulator_name,
            app_name=self.app_name,
        )

        for i in range(3):
            if self.appium_manager.start_appium_server():
                if self.appium_manager.connect(
                    self.device_name, self.string_activities, self.app_package
                ):
                    self.action_extractor = ActionExtractor(
                        self.appium_manager.driver,
                        log_dir=self.master_log_dir,
                        emulator_name=self.emulator_name,
                        app_name=self.app_name,
                    )
                    self.logger.info("Appium server started and connected successfully")
                    # Activate the app for coverage measurement
                    activate_args = self.acv_parser.parse_args(
                        [
                            "activate",
                            self.app_package,
                        ]
                    )
                    run_actions(self.acv_parser, args=activate_args)
                    return True
                else:
                    self.logger.error("Failed to connect to Appium server")
            else:
                self.logger.error("Failed to start Appium server")
        raise Exception(
            "Failed to start Appium server after multiple attempts. Please check the Appium setup."
        )

    def check_emulator_status(self) -> bool:
        return self.emulator_controller.check_emulator_status(self.device_name)

    def cleanup_emulator(self):
        self.emulator_controller.cleanup_emulator(self.device_name)
        self.appium_manager.cleanup_appium()
        self.action_extractor = None  # Reset action extractor

    def perform_action(self, action: Dict[str, Any]):
        # Perform the specified action on the app's GUI.
        # This method handles touch, gesture, text input, and system actions.
        # Action via Appium and ADB commands.

        try:
            if action["type"] == "touch":
                widget_index = action["widget_index"]
                gui_hierarchy = self.appium_manager.driver.page_source
                root = ET.fromstring(gui_hierarchy.encode("utf-8"))
                for element in root.findall(".//*[@clickable='true']"):
                    widget_key = (element.get("class"), element.get("text", ""))
                    if self.gui_embedder.widget_dict.get(widget_key) == widget_index:
                        xpath = self.action_extractor._generate_xpath(element)
                        self.appium_manager.driver.find_element(
                            by="xpath", value=xpath
                        ).click()
                        break
            elif action["type"] == "gesture":
                widget_index = action["widget_index"]
                gesture_type = action["parameters"]["gesture_type"]
                direction = action["parameters"]["direction"]
                if gesture_type == "swipe":
                    gui_hierarchy = self.appium_manager.driver.page_source
                    root = ET.fromstring(gui_hierarchy.encode("utf-8"))
                    for element in root.findall(".//*[@scrollable='true']"):
                        widget_key = (element.get("class"), element.get("text", ""))
                        if (
                            self.gui_embedder.widget_dict.get(widget_key)
                            == widget_index
                        ):
                            bounds = element.get("bounds", "[0,0][0,0]")
                            x, y = self.action_extractor._parse_bounds(bounds)
                            end_x = (
                                x + 500
                                if direction == "right"
                                else x - 500
                                if direction == "left"
                                else x
                            )
                            end_y = (
                                y + 500
                                if direction == "down"
                                else y - 500
                                if direction == "up"
                                else y
                            )
                            self.appium_manager.driver.swipe(
                                start_x=x,
                                start_y=y,
                                end_x=end_x,
                                end_y=end_y,
                                duration=100,
                            )
                            break
            elif action["type"] == "text_input":
                widget_index = action["widget_index"]
                text = action["parameters"]["text"]
                gui_hierarchy = self.appium_manager.driver.page_source
                root = ET.fromstring(gui_hierarchy.encode("utf-8"))
                for element in root.findall(".//*[@class='android.widget.EditText']"):
                    widget_key = (element.get("class"), element.get("text", ""))
                    if self.gui_embedder.widget_dict.get(widget_key) == widget_index:
                        xpath = self.action_extractor._generate_xpath(element)
                        self.appium_manager.driver.find_element(
                            by="xpath", value=xpath
                        ).send_keys(text)
                        break
            elif action["type"] == "system":
                system_action = action["action"]
                if system_action == "toggle_network":
                    subprocess.run(
                        [
                            "adb",
                            "shell",
                            "svc",
                            "wifi",
                            "enable"
                            if self.gui_embedder.get_network_status() == 0
                            else "disable",
                        ]
                    )
                elif system_action == "rotate_screen":
                    subprocess.run(
                        [
                            "adb",
                            "shell",
                            "settings",
                            "put",
                            "system",
                            "user_rotation",
                            "1"
                            if self.gui_embedder.get_device_orientation() == 0
                            else "0",
                        ]
                    )
            self.logger.debug(f"Performed action: {action['type']}")
        except Exception as e:
            self.logger.error(f"Failed to perform action {action}: {str(e)}")

    def save_to_txt(self, content: str, filename: str, directory: str = ""):
        """
        Save content to a text file in the master log directory.
        """
        save_dir = os.path.join(self.master_log_dir, directory)
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, filename)
        try:
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(content)
            self.logger.info(f"Content saved to {file_path}")
        except UnicodeEncodeError as e:
            self.logger.error(
                f"Unicode encoding error when saving to {file_path}: {str(e)}"
            )
            # Try to save with error handling for problematic characters
            with open(file_path, "w", encoding="utf-8", errors="replace") as file:
                file.write(content)
            self.logger.warning(
                f"Content saved to {file_path} with character replacement"
            )

    def save_coverage_report(self):
        try:
            # 5 Make a snap [snap <package_name>]
            snap_args = self.acv_parser.parse_args(
                [
                    "snap",
                    self.app_package,
                    "--wd",
                    self.acv_workdir,
                ]
            )
            run_actions(self.acv_parser, args=snap_args)

            # 6 Apply the extracted coverage data onto the smali code tree [cover-pickles <package_name> --wd <working_dir>]
            cover_pickles_args = self.acv_parser.parse_args(
                [
                    "cover-pickles",
                    self.app_package,
                    "--wd",
                    self.acv_workdir,
                ]
            )
            run_actions(self.acv_parser, args=cover_pickles_args)

            # 7 Generate the code coverage report [report <package_name> --wd <working_dir>]
            report_args = self.acv_parser.parse_args(
                [
                    "report",
                    self.app_package,
                    "--wd",
                    self.acv_workdir,
                ]
            )
            run_actions(self.acv_parser, args=report_args)
            if os.path.exists(self.acv_workdir+"/report"):
                # rename report folder to avoid overwriting
                new_report_dir = f"{self.acv_workdir}/report_{self.ver}"
                os.rename(self.acv_workdir+"/report", new_report_dir)
                self.logger.info(f"Code coverage report generated in {new_report_dir}")
                return
        except Exception as e:
            self.logger.error(f"Failed to generate code coverage report: {str(e)}")

    def get_current_codecov_2_logcat(self):
        # run dump logcat after this
        try:
            calculate_args = self.acv_parser.parse_args(
                [
                    "calculate",
                    "-d",
                    self.device_name,
                    self.app_package,
                ]
            )
            run_actions(self.acv_parser, args=calculate_args)
        except Exception as e:
            self.logger.error(f"Failed to parse ACVTool arguments: {str(e)}")
            return None
    def start_emu_and_appium(self):
        ## Start the emulator and Appium server
        if self.start_emulator():
            self.logger.info(f"Emulator {self.emulator_name} started successfully")
        else:
            sys.exit("Failed to start emulator. Please check the emulator setup.")
        time.sleep(1)  # Wait for emulator to stabilize
        if self.start_appium():
            self.logger.info("Appium server started successfully")
        else:
            sys.exit("Failed to start Appium server. Please check the Appium setup.")
            
    def run_testing(self, max_steps: int = 3, episodes: int = 3):
        """
        MAIN TESTING LOOP
        Run the RL-based GUI testing loop
        """
        ## ATTEMPT TO LOAD PREVIOUS MODEL AND REPLAY BUFFER
        if os.path.exists(self.model_path):
            self.dqn_agent.load_model(self.model_path)
        if os.path.exists(self.replay_buffer_path):
            self.dqn_agent.load_replay_buffer(self.replay_buffer_path)

        self.start_emu_and_appium()
        ## Begin the testing loop
        try:
            for episode in range(episodes):
                self.logger.info(f"Starting episode #{episode}")
                self.logcat_extractor.clear_logcat() # Clear logcat before each episode
                for step in range(max_steps):

                    if not self.check_emulator_status():
                        self.logger.error("Emulator offline, attempting to restart")
                        self.cleanup_emulator()
                        self.start_emu_and_appium()

                    time.sleep(0.5)  # Wait for GUI stability
                    gui_hierarchy = self.appium_manager.driver.page_source
                    current_activity = self.appium_manager.driver.current_activity
                    

                    state = self.gui_embedder.embed(gui_hierarchy, current_activity)
                    # Visualize the extracted state for debugging
                    state_visualization = self.gui_embedder.visualize_state(state, show_details=True)
                    self.logger.info(f"Extracted State:\n{state_visualization}")
                    
                    self.save_to_txt(
                        content=gui_hierarchy,
                        filename=f"gui_hierarchy_E{episode}s{step}_{self.emulator_name}_{self.app_name}_{self.ver}.txt",
                        directory="gui_hierarchies",
                    )
                    # Also save the state visualization
                    self.save_to_txt(
                        content=state_visualization,
                        filename=f"state_visualization_E{episode}s{step}_{self.emulator_name}_{self.app_name}_{self.ver}.txt",
                        directory="gui_hierarchies",
                    )
                    
                    # self.logger.debug(f"widget_dict before extract_actions: {self.gui_embedder.widget_dict}")
                    actions, action_vectors = self.action_extractor.extract_actions(
                        gui_hierarchy, self.gui_embedder.widget_dict
                    )
                    
                    # Log the extracted actions in a more readable format
                    action_summary = []
                    action_summary.append(f"Extracted {len(actions)} possible actions:")
                    for i, action in enumerate(actions):  # Show first 10 actions
                        action_type = action.get('type', 'unknown')
                        if action_type == 'touch':
                            widget_index = action.get('widget_index', -1)
                            # Try to find the widget description
                            widget_desc = "unknown"
                            for widget_key, idx in self.gui_embedder.widget_dict.items():
                                if idx == widget_index:
                                    widget_type, widget_text = widget_key
                                    widget_desc = f"{widget_type.split('.')[-1]}: '{widget_text[:20]}'"
                                    break
                            action_summary.append(f"  {i+1}. Touch {widget_desc}")
                        elif action_type == 'gesture':
                            gesture_type = action.get('parameters', {}).get('gesture_type', 'unknown')
                            direction = action.get('parameters', {}).get('direction', 'unknown')
                            action_summary.append(f"  {i+1}. Gesture: {gesture_type} {direction}")
                        elif action_type == 'text_input':
                            text = action.get('parameters', {}).get('text', '')
                            action_summary.append(f"  {i+1}. Text Input: '{text}'")
                        elif action_type == 'system':
                            sys_action = action.get('action', 'unknown')
                            action_summary.append(f"  {i+1}. System: {sys_action}")
                        else:
                            action_summary.append(f"  {i+1}. {action_type}: {str(action)[:50]}")
                    
                    
                    action_summary_text = "\n".join(action_summary)
                    self.logger.info(f"Action Summary:\n{action_summary_text}")

                    self.save_to_txt(
                        content=f"{action_summary_text}\n\nRaw Data:\n[{str(actions)}, {str(action_vectors)}]",
                        filename=f"actions_E{episode}s{step}_{self.emulator_name}_{self.app_name}_{self.ver}.txt",
                        directory="gui_hierarchies",
                    )

                    if not actions:
                        self.logger.info("No actions available, resetting app")
                        self.restart_app()
                        reward = -10.0
                        self.dqn_agent.store(state, -1, reward, state, True, None)
                        continue
                    action_idx = self.dqn_agent.act(state, actions, action_vectors)
                    action = actions[action_idx]

                    # Clear logs before action and check for crashes after
                    if self.logcat_extractor:
                        self.logcat_extractor.clear_logcat()
                    self.perform_action(action)
                    time.sleep(0.5)

                    # Check for crashes only if logcat_extractor is initialized
                    crash_logs = []
                    if self.logcat_extractor:
                        logs = self.logcat_extractor.dump_logcat() # dump logcat 
                        crash_logs = self.logcat_extractor.extract_crash_logs(logs)
                    has_crash = len(crash_logs) > 0

                        
                    app_left = not self.check_app_status()

                    if app_left and not has_crash:
                        self.restart_app()
                        next_state = self.gui_embedder.embed(
                            self.appium_manager.driver.page_source
                        )
                        next_gui_hierarchy = self.appium_manager.driver.page_source
                    elif not has_crash:
                        next_gui_hierarchy = self.appium_manager.driver.page_source
                        next_state = self.gui_embedder.embed(next_gui_hierarchy)
                    else:
                        self.logger.warning("Crash occurred, restarting app")
                        self.logger.warning(f"Action do crash: {action['type']}")
                        self.logger.warning(
                            f"Location: {action['widget_index']}, {action['parameters']}"
                        )
                        self.logger.warning(f"Crash logs: {"\n###### CRASH/ERROR DETECTED #######\n".join(crash_logs)}")

                        self.restart_app()
                        next_state = None
                        next_gui_hierarchy = None

                    # Extract next action vectors
                    next_action_vectors = (
                        self.action_extractor.extract_actions(
                            next_gui_hierarchy, self.gui_embedder.widget_dict
                        )[1]
                        if next_gui_hierarchy
                        else []
                    )
                    reward = self.reward_analyzer.calculate_reward(
                        state,
                        next_state,
                        action,
                        has_crash,
                        crash_logs,
                        app_left,
                        next_gui_hierarchy,
                    )
                    if next_state is not None and len(next_action_vectors) > action_idx:
                        intrinsic_loss = self.dqn_agent.icm(
                            state, next_action_vectors[action_idx], next_state
                        )
                        intrinsic = intrinsic_loss.item()
                        reward += self.dqn_agent.beta * intrinsic
                    done = step == max_steps - 1

                    # self.logger.debug(f"State: {state}, Action: {action}, Reward: {reward}, Next State: {next_state}, Done: {done}")
                    # self.logger.debug(f"Action vectors: {action_vectors}")
                    
                    self.dqn_agent.store(
                        state, action_idx, reward, next_state, done, next_action_vectors
                    )

                    self.dqn_agent.train()
                    if (
                        len(self.dqn_agent.memory) % 8 == 1
                        and len(self.dqn_agent.memory) > 100
                    ):
                        mmodel = train_macro_generator(
                            self.dqn_agent.memory,
                            self.state_dim,
                            self.action_vector_dim,
                        )
                        self.dqn_agent.macro_generator.load_state_dict(
                            mmodel.state_dict()
                        )

                   
                    self.logger.warning(
                        f"Episode {episode}, Step {step}: Action {str(action['type'])}"
                    )
                    self.logger.warning(
                        f"Reward: {reward}, Crash: {has_crash}, App left: {app_left}"
                    )
            self.get_current_codecov_2_logcat()
            # Save graph and generate HTML visualization at the end of each episode
            # self.dqn_agent.stgraph.save_graph(episode)
        #         self.dqn_agent.save_model(self.model_path)
        #         self.dqn_agent.save_replay_buffer(self.replay_buffer_path)
        #         self.dqn_agent.hidden_state = None

        #         # self.logger.warning(f"Code cov: {self.check_coverage()}")

        finally:
            self.cleanup_emulator()
            self.logger.warning("Testing completed")
