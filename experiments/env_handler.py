import os
import random
import sys
import time
import pathlib as pathlib
import traceback
from selenium.common.exceptions import WebDriverException
from experiments.input_inference import InputSuggestionLLM
from experiments.logger import setup_logger
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.actions.pointer_input import PointerInput
from selenium.webdriver.common.actions.interaction import Interaction
from selenium.webdriver.support.ui import WebDriverWait
import shutil

# from experiments.duel_dqn_2 import DQNAgent, train_macro_generator
from experiments.utils.apk_analyzer import apk_analyzer
from experiments.utils.emulator_n_appium_controller import (
    EmulatorController,
    AppiumManager,
)
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
        self,
        emulator_name: str,
        emulator_port: str,
        appium_port: int,
        apk_path: str,
        master_log_dir: str,
        ver: str,
        txt_generator,
    ):
        self.emulator_name = emulator_name
        self.appium_port = appium_port
        self.emulator_port = emulator_port
        self.apk_path = apk_path
        self.app_name = "-".join(pathlib.Path(apk_path).name.split("."))
        self.acv_intructed_apk_path = ""  # path to the instrumented apk, will define after ACVTool instrumentation
        # Logging setup
        self.ver = ver

        # Create directories for logs and models
        self.master_log_dir = master_log_dir
        self.inst_apk_dir = "apk/instr"
        os.makedirs(self.inst_apk_dir, exist_ok=True)
        self.acv_workdir = "apk/instr/acv_temp"  # will set after apk analyze

        self.logger = setup_logger(
            f"{self.master_log_dir}/env_handler.log",
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

        self.device_name = None  # Will be set after emulator starts
        self.number_of_activities = set()
        self.acv_parser = get_parser()
        self.texts = None
        self.load_texts()
        self.txt_generator = txt_generator

        ###################################### EO__init__
    def load_texts(self):
        ######################################
        #   LOAD TEXT PAYLOADS
        ######################################
        try:
            with open(
                "experiments/test_input.txt", "r"
            ) as file:
                self.texts = [line.strip() for line in file.readlines()]
            if not self.texts:
                self.logger.warning(
                    "No texts found in the file, using default texts."
                )
                self.texts = ["99999", "-999999","9999999999999999999999999", "111.111","-.412","1.1.1.1","@#$%@#!$","test", "user@example.com","@$@#SSAD","123214"]
            self.logger.info(
                f"Loaded {len(self.texts)} texts from file: experiments/test_input.txt"
            )
        except FileNotFoundError:
            self.logger.warning(
                "[FILE NOT FOUND] Text file not found, using default texts"
            )
            self.texts = ["99999", "-999999","9999999999999999999999999", "111.111","-.412","1.1.1.1","@#$%@#!$","test", "user@example.com","@$@#SSAD","123214"]
    
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
            self.number_of_activities = self.exported_activities
            self.logger.info(
                f"APK analysis complete for {self.app_name} ({self.app_package}) - services: {self.services}, receivers: {self.receivers}, providers: {self.providers}, activities: {self.exported_activities}"
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
                #self.cleanup_old_coverage_files() # there may old cov files
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
            sys.exit(0)
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
        for i in range(2):
            try:
                self.logger.info(f"Attempt {i} stopping app: {self.app_package}")
                # Terminate the app if it's running
                self.appium_manager.driver.terminate_app(self.app_package)
                time.sleep(2)  # Let the app close properly
                if not self.check_app_status():
                    self.logger.info("App stopped successfully")
                    return True
                
            except Exception as e:
                self.logger.error(f"Failed to stop app: {str(e)}")
        sys.exit(0)

    def start_app(self) -> bool:
        """Start the app on the emulator."""
        for i in range(2):
            try:
                self.logger.info(f"Attempt {i} starting app: {self.app_package}")
                # Start the app's main activity
                self.appium_manager.driver.activate_app(self.app_package)
                time.sleep(2)  # Let the app stabilize
                self.logger.info("App started successfully")
                return True
            except Exception as e:
                self.logger.error(f"Failed to start app: {str(e)}")
        sys.exit(0)

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
        for i in range(1):
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
                    self.emulator_controller.install_appium_apks()  # have auto retry
                    return True
            except Exception as e:
                self.logger.error(f"Failed to start emulator: {str(e)}")
            time.sleep(2)
        sys.exit(0)

    def start_appium(self) -> bool:
        if not self.analyze_apk():
            self.logger.error("APK analysis failed, cannot start Appium")
            sys.exit(0)
        if not self._intr_apk_w_acvtool():
            self.logger.error("APK instrumentation failed, cannot start test")
            sys.exit(0)

        self.appium_manager = AppiumManager(
            appium_port=self.appium_port,
            apk_path=self.acv_intructed_apk_path,
            log_dir=self.master_log_dir,
            emulator_name=self.emulator_name,
            app_name=self.app_name,
        )

        for i in range(1):
            if self.appium_manager.start_appium_server():
                if self.appium_manager.connect(self.device_name, self.app_package):
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
        sys.exit(0)

    def check_emulator_status(self) -> bool:
        return self.emulator_controller.check_emulator_status(self.device_name)

    def cleanup_emulator(self):
        self.emulator_controller.cleanup_emulator(self.device_name)
        self.appium_manager.cleanup_appium()

    def cleanup_old_coverage_files(self):
        if not os.path.exists(self.acv_workdir):
            self.logger.info(f"ACV workdir {self.acv_workdir} does not exist, skipping cleanup.")
            return
        if os.path.exists(self.acv_workdir + "/ec_files"):
            self.logger.info(f"Cleaning up old ec files in {self.acv_workdir}")
            shutil.move(self.acv_workdir + "/ec_files", self.master_log_dir,)
        if os.path.exists(self.acv_workdir + "/covered_pickles"):
            self.logger.info(f"Cleaning up old covered pickles in {self.acv_workdir}")
            shutil.move(self.acv_workdir + "/covered_pickles", self.master_log_dir)

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
            time.sleep(5)  # Wait for report generation
            if os.path.exists(self.acv_workdir + "/report"):
                # rename report folder to avoid overwriting
                new_report_dir = f"{self.master_log_dir}/report_{self.ver}_{time.strftime('%Y%m%d_%H%M%S')}"
                shutil.move(self.acv_workdir + "/report", new_report_dir)
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
            sys.exit(0)
        time.sleep(1)  # Wait for emulator to stabilize
        if self.start_appium():
            self.logger.info("Appium server started successfully")
        else:
            sys.exit(0)
        return True

    def perform_action(self, action: dict, action_taken_vector: list, ref_act: dict = None):
        """
        Perform an action on the UI using W3C Actions (Appium v4+).
        Logs every step for debugging and RL traceability.
        """
        try:
            idx = action_taken_vector.index(1) if 1 in action_taken_vector else None
            enabled = action.get("status", [False])[0]
            if not enabled:
                self.logger.info(f"[SKIP] Action {idx} skipped: Element not enabled.")
                return None

            res_id = action.get("resource_id")
            pos_norm = action.get("position_norm", [0]*6)
            pos = action.get("position", [0]*6)
            screen = self.appium_manager.driver.get_window_size()
            w, h = screen["width"], screen["height"]
            cx = int(pos_norm[4] * w) if 0 < pos_norm[4] <= 1 else int(pos[4])
            cy = int(pos_norm[5] * h) if 0 < pos_norm[5] <= 1 else int(pos[5])

            # Try finding element by ID
            el = None
            if res_id:
                try:
                    el = self.appium_manager.driver.find_element("id", res_id)
                    if el.get_attribute("class") == action["type"]:
                        self.logger.info(f"[INFO] Element found: {res_id} {action['type']}")
                except NoSuchElementException:
                    self.logger.warning(f"[WARN] Element with resource ID '{res_id}' not found.")

            if not el:
                self.logger.info(f"[INFO] No element found by ID '{res_id}', using coordinates ({cx}, {cy}).")

            # W3C tap implementation
            def do_w3c_tap(target_el=None):
                touch = PointerInput(Interaction.POINTER_TOUCH, "finger")
                actions = ActionChains(self.appium_manager.driver)
                actions.w3c_actions = ActionChains(self.appium_manager.driver).w3c_actions
                actions.w3c_actions.add_pointer_input(touch)
                origin = target_el if target_el else PointerInput.Origin.viewport
                x, y = (target_el.rect["x"] + target_el.rect["width"] // 2,
                        target_el.rect["y"] + target_el.rect["height"] // 2) if target_el else (cx, cy)
                pa = actions.w3c_actions.pointer_action
                pa.create_pointer_move(0, origin, x, y)
                pa.create_pointer_down(0)
                pa.create_pointer_up(0)
                actions.perform()

            action_type = None
            log_prefix = f"[ACTION idx={idx} id={action.get('id_hash')}]"
            if idx is None:
                self.logger.warning(f"{log_prefix} Skipping action: No action taken.")
                return None
            
            if idx == 0:  # Click
                try:
                    args = {"elementId": el.id} if el else {"x": cx, "y": cy}
                    self.appium_manager.driver.execute_script("mobile: clickGesture", args)
                except Exception:
                    self.logger.warning(f"{log_prefix} clickGesture failed, fallback to W3C tap.")
                    do_w3c_tap(el)
                action_type = "click"

            elif idx == 1:  # Long click
                args = {"duration": 1000}
                args["elementId"] = el.id if el else None
                if not el:
                    args.update({"x": cx, "y": cy})
                self.appium_manager.driver.execute_script("mobile: longClickGesture", args)
                action_type = "long_click"

            elif idx == 13:  # Double click
                args = {"elementId": el.id} if el else {"x": cx, "y": cy}
                self.appium_manager.driver.execute_script("mobile: doubleClickGesture", args)
                action_type = "double_click"

            elif idx in (2, 3):  # Text input
                if el:
                    el.click()
                else:
                    do_w3c_tap()
                # Wait for focused element
                try:
                    WebDriverWait(self.appium_manager.driver, 3).until(lambda d: d.switch_to.active_element)
                    focused = self.appium_manager.driver.switch_to.active_element
                    if not focused:
                        self.logger.warning(f"{log_prefix} no focused input.")
                        return False
                    context = {
                        "resource_id": action.get("resource_id", "") + action.get("type", "") ,
                        "content_desc": action.get("text_raw", ""),
                        "input_type": action.get("input_type_raw", ""),
                    }
                    self.logger.info(f"context for text input: {context}")
                    suggestions = self.txt_generator.suggest_inputs(context)

                    for txt in suggestions:
                        if not focused.is_displayed():
                            break
                        focused.clear()
                        focused.send_keys(txt + "\n")
                    action_type = "edit_number" if idx == 2 else "edit_text"
                except Exception:
                    self.logger.error(f"{log_prefix} failed to focus input.")
                    return False

            elif idx in (4, 5, 6, 7):  # Scroll gestures
                dirs = {4: "up", 5: "down", 6: "left", 7: "right"}
                direction = dirs[idx]
                args = {"direction": direction, "percent": 0.75}
                if el:
                    args["elementId"] = el.id
                self.appium_manager.driver.execute_script("mobile: swipeGesture", args)
                action_type = f"scroll_{direction}"

            elif idx in (8, 9):  # Rotate
                orientation = "LANDSCAPE" if idx == 8 else "PORTRAIT"
                self.appium_manager.driver.orientation = orientation
                action_type = f"rotate_{orientation.lower()}"

            elif idx in (10, 11):  # Volume
                key = 24 if idx == 10 else 25
                self.appium_manager.driver.press_keycode(key)
                action_type = "volume_up" if idx == 10 else "volume_down"

            elif idx == 12:  # Back
                self.appium_manager.driver.press_keycode(4)
                action_type = "back"

            self.logger.info(f"{log_prefix} Executed action: {action_type} at ({cx}, {cy})")
            return action_type

        except Exception as e:
            self.logger.error("[ERROR] perform_action failed", exc_info=True)
            self.logger.error(traceback.print_exc())
            return None
