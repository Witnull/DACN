import datetime
import pathlib
import signal
import sys
import threading
import traceback

from matplotlib import pyplot as plt
import numpy as np
import torch
from experiments.env_handler import EnviromentHandler
import os
import time
from experiments.gui_embedder import GUIEmbedder
from experiments.logger import setup_logger
from experiments.logcat_extractor import LogcatExtractor
from experiments.reward_analyzer import RewardAnalyzer
from experiments.duel_dqn_agent import DQNAgent, MetaDQNAgent
from experiments.state_embedder import StateEmbedder
from experiments.utils.path_config import ADB_PATH

log_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
global interrupted
interrupted = False  # Flag for detection


def handle_sigint(signum, frame):
    global interrupted
    print("\n[INFO] SIGINT received. Gracefully shutting down...")
    interrupted = True


signal.signal(signal.SIGINT, handle_sigint)


class TestController:
    def __init__(
        self,
        emulator_name="avd002",
        appium_port=4723,
        emulator_port=5554,
        apk_path="./apk/*.apk",
        txt_generator=None,  # Placeholder for text generator
    ):
        self.emulator_name = emulator_name
        self.appium_port = appium_port
        self.emulator_port = emulator_port
        self.apk_path = apk_path
        self.app_name = "-".join(pathlib.Path(apk_path).name.split("."))

        self.adb_path = ADB_PATH

        # Logging setup
        self.ver = "t" + str(log_time)  # Timestamp for versioning log

        # Create directories for logs and models
        self.master_log_dir = f"Logs/{self.emulator_name}_{self.app_name}_{self.ver}"
        os.makedirs(self.master_log_dir, exist_ok=True)
        self.model_dir = (
            f"results/model_{self.emulator_name}_{self.app_name}_{self.ver}"
        )
        os.makedirs(self.model_dir, exist_ok=True)

        # Set saving paths
        self.model_path = f"{self.model_dir}/model_{self.ver}.pth"
        self.replay_buffer_path = f"{self.model_dir}/replay_buffer_{self.ver}.pkl"

        self.logger = setup_logger(
            f"{self.master_log_dir}/test_controller.log",
            emulator_name=self.emulator_name,
            app_name=self.app_name,
        )
        self.logger.info(
            f"Testing Controller initialized for {self.emulator_name} and {self.app_name}"
        )

        self.env_handler = EnviromentHandler(
            self.emulator_name,
            self.emulator_port,
            self.appium_port,
            self.apk_path,
            self.master_log_dir,
            self.ver,
            txt_generator=txt_generator,  # Placeholder for text generator
        )
        # Initialize components
        self.gui_embedder = GUIEmbedder(
            log_dir=self.master_log_dir,
            emulator_name=self.emulator_name,
            app_name=self.app_name,
        )
        self.reward_analyzer = RewardAnalyzer(
            log_dir=self.master_log_dir,
            emulator_name=self.emulator_name,
            app_name=self.app_name,
        )

        self.state_embedder = None
        self.dqn_agent = None
        self.device_name = None
        self.logcat_extractor = None
        self.appium_manager = None
        # TESTING CONFIG ############################################
        self.condition_modes = [
            "random_navigate",
            "fill_input",
            "submit_action",
            "go_back",
            "system_action",
        ]
        self.action_space_dim = self.gui_embedder.action_space_dim
        # Is the value of the edge
        self.state_embedder_feature_dim = self.action_space_dim
        self.state_embedder_output_dim = 96
        self.test_hrl = False
        self.conditioned_state_dim = self.state_embedder_output_dim
        if self.test_hrl:
            self.conditioned_state_dim = self.state_embedder_output_dim + len(
                self.condition_modes
            )

        self.logger.info(
            f"State embedder feature dim: {self.state_embedder_feature_dim},\noutput dim: {self.state_embedder_output_dim},\naction space dim: {self.action_space_dim}"
        )

        self.number_of_activities = None
        self.found_activities = set()
        self.instr_cov = 0.0
        self.activity_cov = 0.0
        self.crash_or_error_logs = []
        self.episode = 0
        self.step = 0

    def snapshot_gui(
        self,
        prev_action_vector,
        prev_selected_action_space_vector_tensor,
        prev_activity_id_hash,
        prev_elm_id_hash,
    ):
        try:
            gui_hierarchy = self.appium_manager.driver.page_source
            current_activity = self.appium_manager.driver.current_activity
            screenshot = None
            for i in range(3):
                try:
                    screenshot = self.appium_manager.driver.get_screenshot_as_png()
                    break
                except Exception as e:
                    self.logger.error(f"Error taking screenshot: {str(e)}")
                    screenshot = None
                    time.sleep(2)  # Wait before retrying

            (
                activity_id_hash,
                action_space_vector_tensor,
                possible_actions,
                vis_embeddings_tensor,
            ) = self.gui_embedder.gui_embed(
                gui_hierarchy,
                current_activity,
                screenshot,
                prev_action_vector,
                prev_selected_action_space_vector_tensor,
                prev_activity_id_hash,
                prev_elm_id_hash,
            )
            state_vector_tensor = self.state_embedder.extract_state_vector(
                self.gui_embedder.graph,
            )
            self.found_activities.add(current_activity)
            # self.save_to_txt(
            #     content=gui_hierarchy,
            #     filename=f"gui_hierarchy_E{self.episode}s{self.step}_{self.emulator_name}_{self.app_name}_{self.ver}.txt",
            #     directory="gui_hierarchies",
            # )
            return (
                current_activity,
                state_vector_tensor,
                activity_id_hash,
                action_space_vector_tensor,
                possible_actions,
            )
        except Exception as e:
            self.logger.error(f"Error in GUI snapshot: {str(e)}")
            self.logger.error(traceback.print_exc())
            # Return values consistent with normal return type
            return (
                None,
                torch.zeros(self.gui_embedder.action_space_dim),
                "",
                torch.zeros(0),
                [],
            )

    def meta_action_categorize(
        self, mode, action_vector
    ):  # policy map from action to target mode
        # Meta action categorization
        # mode = [0,0,0,0,1,]
        # action_vector = [0,0,1,0,0,0,0,0,0,0]  # Example action vector
        mode_idx = mode  # int
        action_vector_idx = action_vector.index(1)
        if mode_idx == 0:
            return False  # "random_navigate" mode does not have specific actions
        elif mode_idx == 1:
            if action_vector_idx in [2, 3]:
                return True
        elif mode_idx == 2:
            if action_vector[3] == 1:
                return True
        elif mode_idx == 3:
            if action_vector_idx == 12:
                return True
        elif mode_idx == 4:  # "system"
            if action_vector_idx in [8, 9, 10, 11]:
                return True
        return False

    def initializing_testing(self, batch_size: int = 16):
        self.dqn_agent = DQNAgent(
            log_dir=self.master_log_dir,
            app_name=self.app_name,
            emulator_name=self.emulator_name,
            ver=self.ver,
            state_tensor_dim=self.conditioned_state_dim,
            action_space_tensor_dim=self.action_space_dim,
            action_dim=self.gui_embedder.action_dim,
            lr=1e-3,
            batch_size=batch_size,
        )

        self.state_embedder = StateEmbedder(
            log_dir=self.master_log_dir,
            emulator_name=self.emulator_name,
            app_name=self.app_name,
            feature_dim=self.state_embedder_feature_dim,
            out_dim=self.state_embedder_output_dim,
            action_space_dim=self.action_space_dim,
            action_dim=self.gui_embedder.action_dim,
        )

        self.logcat_extractor = LogcatExtractor(
            adb_path=self.adb_path,
            device_udid=self.device_name,
            app_name=self.app_name,
            logdir=self.master_log_dir,
        )

        self.appium_manager = self.env_handler.appium_manager
        self.number_of_activities = self.env_handler.number_of_activities

        self.meta_agent = MetaDQNAgent(
            input_dim=self.state_embedder_output_dim,
            num_modes=len(self.condition_modes),
        )

    def run_testing(
        self,
        max_steps: int = 16 * 5,
        episodes: int = 5,
        time_limit: int = -1,
        load_data: bool = False,
    ):
        global interrupted
        """
        MAIN TESTING LOOP
        Run the RL-based GUI testing loop
        """
        ### Initialize the environment and appium manager
        success = self.env_handler.start_emu_and_appium()
        if not success:
            self.logger.error(traceback.print_exc())
            sys.exit(0)

        self.device_name = self.env_handler.device_name
        self.logger.info(f"Updated device name: {self.device_name}")
        batch_size = max(4, int(max_steps / 4))
        self.logger.info(f"Batch size set to: {batch_size}")
        self.initializing_testing(batch_size)  # Initialize the testing components
        current_mode = 0
        meta_interval = 4  # Change mode every 4 steps

        ## ATTEMPT TO LOAD PREVIOUS MODEL AND REPLAY BUFFER
        if load_data:
            if os.path.exists(self.model_path):
                self.dqn_agent.load_model(self.model_path)
            if os.path.exists(self.replay_buffer_path):
                self.dqn_agent.load_replay_buffer(self.replay_buffer_path)
        picked_actions = []
        time_start = time.time()
        ## Begin the testing loop
        try:
            ######################
            # PRESET SOME VALUES
            ######################
            prev_activity_id_hash = None  # What activity was before
            prev_action_vector = None  # What action taken
            prev_state_vector_tensor = None  # What state was before
            prev_selected_action_space_vector_tensor = None  # What actions were taken
            prev_reward = 0.0  # What reward was before
            prev_elm_id_hash = None  # Selected element to act
            prev_meta_action = None  # What meta-action was taken  # noqa: F841
            prev_meta_action_vector = (
                None  # What meta-action vector was taken # noqa: F841
            )
            prev_meta_action_reward = (
                0.0  # What meta-action reward was taken # noqa: F841
            )
            test_hrl = self.test_hrl
            done = (
                not test_hrl
            )  # If HRL is not used, set done to True to skip meta-action categorization

            if time_limit > 0:
                self.logger.info(f"Testing will run for {time_limit} minutes")
                episodes = 999_999_999  # Set a high number of episodes to run until time limit is reached

            ##########################
            # EPISODE LOOP
            # Meta action control
            ##########################
            for episode in range(episodes):
                self.episode = episode
                meta_reward = 0.0  # Reset meta reward for each episode
                meta_done = False  # Reset meta done for each episode # noqa: F841
                # Reset meta-action for each episode

                if interrupted:
                    self.logger.warning(
                        "Testing interrupted by user. Exiting gracefully."
                    )
                    sys.exit(0)

                self.logger.info(
                    f"Starting episode #{episode}/{episodes}. Max #{max_steps} steps/eps "
                )
                self.logcat_extractor.clear_logcat()  # Clear logcat before each episode

                if not self.appium_manager.driver or not self.appium_manager.driver.session_id:
                    self.logger.error("Driver is not active")
                    self.appium_manager.restart_appium()

                # Init snapshot of the app GUI
                (
                    current_activity,
                    state_vector_tensor,
                    activity_id_hash,
                    action_space_vector_tensor,
                    possible_actions,
                ) = self.snapshot_gui(
                    prev_action_vector,
                    prev_selected_action_space_vector_tensor,
                    prev_activity_id_hash,
                    prev_elm_id_hash,
                )
                print(f"State vector tensor: {state_vector_tensor.shape}")

                ######################
                # META CONTROLLER MODE SELECTION
                # Select each episode
                # when finish/ reached mode's target
                # break the steping/ set done and move to next eps
                ######################

                if test_hrl:
                    current_mode = self.meta_agent.select_mode(state_vector_tensor)
                    self.logger.info(f"Current mode: {current_mode}")
                    mode_vec = (
                        torch.nn.functional.one_hot(
                            torch.tensor(current_mode),
                            num_classes=len(self.condition_modes),
                        )
                        .float()
                        .to(device=state_vector_tensor.device)
                    )
                    conditioned_state = torch.cat(
                        [state_vector_tensor.squeeze(0), mode_vec], dim=0
                    )
                    self.logger.debug(
                        f"Conditioned state shape: {conditioned_state.shape}"
                    )

                #############################
                # STEPPING
                # Sub action
                #############################
                done = False
                eps_reward = 0.0  # Episode reward
                for step in range(max_steps):
                    if step + 1 == max_steps:
                        done = True  # If max steps reached, set done to True

                    self.step = step
                    reward = 0.0
                    action_failed = False

                    if interrupted:
                        self.logger.warning(
                            f"Stopping testing at {episode} episode, {step}/{max_steps} step"
                        )
                        sys.exit(0)

                    if not self.env_handler.check_emulator_status():
                        ######################
                        # IF THE EMULATOR SHUTDOWN/ EXIT
                        # Attempt to restart the emulator and appium
                        # This is a safety net to ensure the testing continues
                        ######################
                        self.logger.error("Emulator offline, attempting to restart")
                        self.env_handler.cleanup_emulator()
                        self.env_handler.start_emu_and_appium()
                        time.sleep(1)  # Wait for emulator to stabilize
                        (
                            current_activity,
                            state_vector_tensor,
                            activity_id_hash,
                            action_space_vector_tensor,
                            possible_actions,
                        ) = self.snapshot_gui(
                            prev_action_vector,
                            prev_selected_action_space_vector_tensor,
                            prev_activity_id_hash,
                            prev_elm_id_hash,
                        )

                    self.logger.debug(f"Activity ID hash: {activity_id_hash}")
                    self.logger.info(f"Possible actions - Act Space: {len(possible_actions)} - {action_space_vector_tensor.shape}")
                    # self.logger.debug(possible_actions)
                    # self.logger.debug(action_space_vector_tensor)
                    conditioned_state = state_vector_tensor

                    ######################
                    # SELECT AND DO ACTION
                    ######################

                    action_taken_idx, action_taken_vector = (
                        self.dqn_agent.select_action(
                            conditioned_state,
                            action_space_vector_tensor,
                            possible_actions,
                        )
                    )
                    action_taken = possible_actions[action_taken_idx]
                    # Clear logs before action and check for crashes after
                    if self.logcat_extractor:
                        self.logcat_extractor.clear_logcat()
                    # Perform the action in the app
                    action_type = self.env_handler.perform_action(
                        action_taken,
                        action_taken_vector,
                        self.gui_embedder.action_types,
                    )

                    self.logger.info(
                        f"Action taken: {action_type}, Action vector: {action_taken_vector}"
                    )
                    if not action_type:
                        done = True  # If action failed, mark as done
                        action_failed = True
                        self.logger.error(
                            f"Action {action_taken} failed, marking episode as done"
                        )
                    else:
                        picked_actions.append(action_taken_vector)
                    time.sleep(0.3)

                    # Check if the action was in scope of the expected modes
                    if (
                        self.meta_action_categorize(current_mode, action_taken_vector)
                        and test_hrl
                    ):
                        self.logger.info(
                            f"Action {action_taken} is in scope of the expected mode {current_mode}"
                        )
                        # meta_reward = (
                        #     eps_reward - prev_reward
                        # )  # Cumulative reward since last meta-step
                        done = True

                    ######################
                    # ADD ACTION TO MEMORY
                    ######################

                    if any(
                        x is None
                        or not isinstance(x, torch.Tensor)
                        or torch.isnan(x).any()
                        or torch.isinf(x).any()
                        or x.numel() == 0
                        or x.eq(0).all()  # All zeros = placeholder
                        for x in [
                            prev_state_vector_tensor,
                            prev_selected_action_space_vector_tensor,
                            conditioned_state,
                            action_space_vector_tensor,
                        ]
                    ):
                        self.logger.error(
                            "Skipping invalid transition, Due to invalid tensor values..."
                        )
                    elif any(
                        a is None
                        or sum(a) == 0  # All zeros = placeholder
                        or not isinstance(a, list)
                        for a in [
                            prev_action_vector,
                        ]
                    ):
                        self.logger.error(
                            "Skipping invalid transition, Due to invalid action vector values..."
                        )
                    elif prev_reward <= -25.0:
                        self.logger.warning(
                            f"Reward is too low {prev_reward}, skipping add to memory..."
                        )
                    else:
                        self.dqn_agent.memory.push(
                            prev_state_vector_tensor.squeeze(0),  # [feat]
                            prev_selected_action_space_vector_tensor,  # [feat]
                            prev_action_vector,  # array
                            prev_reward,  # float
                            conditioned_state.squeeze(0),  # [feat]
                            action_space_vector_tensor[action_taken_idx],  # [feat]
                            done,  # bool
                        )
                        if (step + 1) % 8 == 0:
                            self.dqn_agent.memory.export_to_csv(
                                f"{self.master_log_dir}/replay_buffer_{self.ver}.csv"
                            )
                            self.logger.info(
                                f"Replay buffer saved to {self.master_log_dir}/replay_buffer_{self.ver}.csv"
                            )

                    ######################
                    # SAVE TO PREV_ VARIABLES
                    # Avoid overwriting them in the next iteration
                    ######################
                    prev_activity_id_hash = activity_id_hash
                    prev_elm_id_hash = action_taken.get("id_hash", None)
                    prev_action_vector = action_taken_vector
                    prev_selected_action_space_vector_tensor = (
                        action_space_vector_tensor[action_taken_idx]
                    )
                    prev_state_vector_tensor = conditioned_state

                    ######################
                    # EXTRACT LOGS AND CHECK FOR CRASHES/ERRORS FOUND
                    ######################
                    crash_or_error_logs = []
                    if self.logcat_extractor:
                        logs = self.logcat_extractor.dump_logcat()  # dump logcat
                        crash_or_error_logs = self.logcat_extractor.extract_logs(logs)
                        # Check if there are duplicate crash logs
                        crash_or_error_logs = list(set(crash_or_error_logs))

                    has_crash_or_error = len(crash_or_error_logs) > 0
                    app_left = not self.env_handler.check_app_status()

                    if app_left:
                        self.env_handler.restart_app()

                    if has_crash_or_error:
                        for log in crash_or_error_logs:
                            if log not in self.crash_or_error_logs:
                                self.crash_or_error_logs.append(log)
                                self.logger.warning(
                                    f"Crash logs: {'\n###### CRASH/ERROR DETECTED #######\n'.join(crash_or_error_logs)}"
                                )
                    ######################
                    # MOVE TO NEXT STATE
                    ######################
                    self.found_activities.add(current_activity)
                    (
                        current_activity,
                        state_vector_tensor,
                        activity_id_hash,
                        action_space_vector_tensor,
                        possible_actions,
                    ) = self.snapshot_gui(
                        prev_action_vector,
                        prev_selected_action_space_vector_tensor,
                        prev_activity_id_hash,
                        prev_elm_id_hash,
                    )

                    ######################
                    # GET INSTR COVERAGE AND ACTIVITY COVERAGE
                    ######################
                    self.env_handler.get_current_codecov_2_logcat()
                    logs = self.logcat_extractor.dump_logcat()
                    instr_cov = self.logcat_extractor.extract_acv_coverage(logs)
                    self.instr_cov = (
                        instr_cov if (instr_cov > self.instr_cov) else self.instr_cov
                    )  # prevent fault cov
                    self.activity_cov = (
                        (len(self.found_activities) / len(self.number_of_activities))
                        if len(self.number_of_activities) > 0
                        else 0
                    )
                    ######################
                    # CALC REWARD
                    ######################
                    reward += self.reward_analyzer.calculate_reward(
                        (time.time() - time_start) / 60,
                        self.found_activities,
                        self.instr_cov,
                        self.activity_cov,
                        prev_activity_id_hash,
                        prev_elm_id_hash,
                        activity_id_hash,
                        state_vector_tensor,
                        has_crash_or_error,
                        crash_or_error_logs,
                        app_left,
                        action_failed,
                    )
                    prev_reward = reward
                    eps_reward += reward

                    if not test_hrl and step % batch_size == 0:
                        self.dqn_agent.train_replay()

                    self.logger.warning(
                        f"\n\n==============\n\nTime running: {(time.time() - time_start) / 60:.2f}\n==Episode: {episode}\n==Step: {step}\n==Reward: {reward:.2f}\n==Code cov: {self.instr_cov * 100:.5f}%\n==Activity cov: {self.activity_cov * 100:.5f}%\n==Found activities: {len(self.found_activities)} / {len(self.number_of_activities)}\n==Possible found errors:{len(self.crash_or_error_logs)}\n==Crash: {has_crash_or_error}\n==App left: {app_left}\n\n==============\n\n"
                    )
                    if (time.time() - time_start) / 60 > time_limit > 0:
                        self.logger.info(f"Time limit reached: {time_limit} minutes")
                        break

                # self.gui_embedder.graph_dataset_save()
                self.dqn_agent.train_replay()
                if episode % 2 == 0:
                    threading.Thread(target=self.env_handler.save_coverage_report).start()
                self.save_testing_data(picked_actions)
                self.save_to_txt(f"{eps_reward}\n", "episode_rewards.txt", "a")

                self.logger.warning(
                    f"Episode {episode} completed.\n==Episode reward: {eps_reward:.2f}"
                )
                self.dqn_agent.save_training_history(
                    f"{self.master_log_dir}/training_history_{self.ver}.csv"
                )

                if (time.time() - time_start) / 60 > time_limit > 0:
                    self.logger.warning(
                        f"Time limit reached: {(time.time() - time_start) / 60}/{time_limit} minutes"
                    )
                    break

                if test_hrl:
                    self.meta_agent.replay_buffer.push(
                        prev_state_vector_tensor,  # state vector
                        None,
                        current_mode,  # selected mode as 1hot
                        meta_reward,  # computed manually
                        state_vector_tensor,
                        None,
                        True,
                    )
                    self.meta_agent.train_step()
                    self.meta_agent.update_target()

        except Exception as e:
            self.logger.error(f"An error occurred during testing: {str(e)}")
            self.logger.error(traceback.print_exc())
        finally:
            self.logger.warning(
                f"\n\n====== COMPLETE TESTING ========\n\n==Time running: {(time.time() - time_start) / 60:.2f}\n==Episode {self.episode}\n==Step {self.step}\n==Code cov: {self.instr_cov * 100:.5f}%\n==Activity cov: {self.activity_cov * 100:.5f}%\n==Found activities: {len(self.found_activities)} / {len(self.number_of_activities)}\n==Possible found errors:{len(self.crash_or_error_logs)}\n\n==============\n\n"
            )
            self.logger.warning(
                "\n\n====== ACTIVITIES FOUND ========\n"
                "\n========\n"
                f"{
                    '\n'.join(self.found_activities)
                    if self.found_activities
                    else 'No activities found'
                }"
                "\n========\n"
            )
            self.logger.warning(
                "\n\n====== CRASH/ERROR FOUND ========\n"
                "\n========\n"
                f"{
                    '\n'.join(self.crash_or_error_logs)
                    if self.crash_or_error_logs
                    else 'No crash/error logs'
                }"
                "\n========\n"
            )
            self.env_handler.save_coverage_report()
            save_done = self.save_testing_data(picked_actions)
            time.sleep(35)  # Give some time for the app to stabilize before cleanup
            if (
                not interrupted and save_done  # Ctrl + C check
            ):  # If interrupted, don't clean up cuz it already cleaned up
                self.env_handler.cleanup_emulator()
                time.sleep(30)  # Give some time for the app to stabilize before cleanup
                self.env_handler.cleanup_old_coverage_files()  # prevent old coverage files from accumulating
            self.logger.warning("Testing completed")
            return True

    def save_testing_data(self, picked_actions):
        """
        Save testing data to the master log directory.
        """
        self.dqn_agent.save_model(self.model_path)
        self.dqn_agent.save_replay_buffer(self.replay_buffer_path)

        # Save plot:
        # Count frequencies per action
        picked_actions_np = np.array(picked_actions)
        action_counts = picked_actions_np.sum(axis=0)

        # Map to action names
        action_labels = [
            name
            for name, idx in sorted(
                self.gui_embedder.action_types.items(), key=lambda x: x[1]
            )
        ]

        # Plot
        plt.figure(figsize=(10, 5))
        bars = plt.bar(action_labels, action_counts, color="skyblue")
        plt.title("Action Distribution")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.grid(True, axis="y", linestyle="--", alpha=0.5)

        # Add labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

        plt.savefig(f"{self.master_log_dir}/action_distribution_plot.png")

        self.logger.info(f"action_space_dim: {self.action_space_dim}")
        self.logger.info(
            f"state_embedder_feature_dim: {self.state_embedder_feature_dim}"
        )
        self.logger.info(f"state_embedder_output_dim: {self.state_embedder_output_dim}")
        self.logger.info(f"Graph nodes: {self.gui_embedder.graph.number_of_nodes()}")
        self.logger.info(f"Graph edges: {self.gui_embedder.graph.number_of_edges()}")
        return True

    def save_to_txt(
        self, content: str, filename: str, mode: str = "a", directory: str = ""
    ):
        """
        Save content to a text file in the master log directory.
        """
        save_dir = os.path.join(self.master_log_dir, directory)
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, filename)
        try:
            with open(file_path, mode, encoding="utf-8") as file:
                file.write(content)
            self.logger.info(f"Content saved to {file_path}")
        except UnicodeEncodeError as e:
            self.logger.error(
                f"Unicode encoding error when saving to {file_path}: {str(e)}"
            )
            # Try to save with error handling for problematic characters
            with open(file_path, mode, encoding="utf-8", errors="replace") as file:
                file.write(content)
            self.logger.warning(
                f"Content saved to {file_path} with character replacement"
            )
