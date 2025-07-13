import hashlib
import traceback
import torch
import numpy as np
from typing import Set, Tuple,Dict
from scipy.spatial.distance import cosine
from experiments.logger import setup_logger

REWARDS = {
    "crash_or_err": 10,
    "repeated_crash_or_err": 1,
    "new_activity": 10,
    "new_state": 5,
    "repeated_state": -1,
    "new_transition": 10,
    "app_left": -5,
    "default_penalty": -1,
    "failed_action": -10,
    "cov":10,
}

class RewardAnalyzer:
    def __init__(
        self,
        log_dir: str = "app/logs/",
        emulator_name: str = "Unknown",
        app_name: str = "Unknown",
        similarity_threshold: float = 0.85,
        familiar_threshold: float = 0.98,
        transition_weight: float = 0.5,
    ):
        self.logger = setup_logger(
            f"{log_dir}/reward_analyzer.log",
            emulator_name=emulator_name,
            app_name=app_name,
        )
        self.seen_activities = set()
        self.seen_activities_widgets: Dict[str, Set[str]] = {}  # fixed
        self.seen_states = {}
        self.seen_crashes = set()
        self.raw_crashes = []
        self.seen_transitions: Dict[Tuple[str, str], Set[Tuple[str, str, str]]] = {}
        self.similarity_threshold = similarity_threshold
        self.familiar_threshold = familiar_threshold
        self.max_time = 60 * 60 * 3
        self.transition_weight = transition_weight
        self.step_count = 0
        self.instr_cov = 0.0
        self.activity_cov = 0.0

    # def calculate_reward(
    #     self,
    #     time_elapsed: float,
    #     found_activities,
    #     instr_cov: float,
    #     activity_cov: float,
    #     prev_activity_id_hash: str,
    #     prev_elm_id_hash: str,
    #     activity_id_hash: str,
    #     state_vector: torch.Tensor,
    #     has_crash: bool,
    #     crash_or_error_logs,
    #     app_left: bool,
    #     action_failed: bool
    # ) -> float:
    #     try:
    #         reward = 0.0

    #         if action_failed:
    #             return REWARDS["failed_action"]
            
    #         if app_left and not has_crash:
    #             self.logger.warning("App left detected, applying penalty")
    #             return REWARDS["app_left"]

    #         # Handle crash
    #         if has_crash:
    #             new_crash_or_error = False
    #             for crash_or_error_log in crash_or_error_logs:
    #                 csh = hashlib.md5(crash_or_error_log.strip().encode()).hexdigest()
    #                 if csh not in self.seen_crashes:
    #                     new_crash_or_error = True
    #                     self.seen_crashes.add(csh)
    #                     self.raw_crashes.append(crash_or_error_log)
    #             reward += REWARDS["crash"] if new_crash_or_error else REWARDS["repeated_crash"]
    #             self.logger.info(f"{'New' if new_crash_or_error else 'Repeated'} crash: reward += {reward}")

    #         # Base reward: new activity
    #         base_reward = 0.0
    #         new_activity = found_activities.difference(self.seen_activities)
    #         if new_activity:
    #             base_reward += REWARDS["new_activity"]
    #             self.logger.info(f"New activity {new_activity} found: reward += {REWARDS['new_activity']}")    
    #         self.seen_activities |= found_activities
            
    #         # Similarity calculation
    #         state_vector = state_vector.cpu().numpy()
    #         key = activity_id_hash + "_" + prev_elm_id_hash
    #         if key not in self.seen_states:
    #             self.seen_states[key]=[]
    #             self.seen_states[key].append((state_vector))
    #             self.logger.info(f"New state added: {key}")
    #             base_reward += REWARDS["new_state"]
    #         else:
    #             max_similarity = 0.0
    #             for seen_vector in self.seen_states[key]:
    #                 try:
    #                     u = state_vector.flatten()
    #                     v = seen_vector.flatten()
    #                     norm_u = np.linalg.norm(u)
    #                     norm_v = np.linalg.norm(v)

    #                     if norm_u == 0 or norm_v == 0:
    #                         max_similarity = 1  # or define a default 1 # old state
    #                         self.logger.warning(f"One of the vectors is zero, setting similarity to 1.0\n state_vector: {u}, seen_vector: {v}")
    #                     else:
    #                         max_similarity = 1 - cosine(u, v)
    #                 except Exception as e:
    #                     self.logger.warning(f"Similarity comparison failed: {e}")
    #                     self.logger.error(traceback.print_exc())
    #                     continue

    #             if max_similarity < self.similarity_threshold:
    #                 base_reward += REWARDS["new_state"]
    #                 self.seen_states[key].append((state_vector))
    #             elif max_similarity > self.familiar_threshold:
    #                 base_reward += REWARDS["repeated_state"]
    #             else:
    #                 base_reward += ((1 - max_similarity) * REWARDS["new_state"])

    #             self.logger.info(f"State similarity: {max_similarity:.2f}, base_reward now {base_reward:.2f}")


    #         # State widget exploration
    #         state_reward = 0.0
    #         if activity_id_hash not in self.seen_activities_widgets:
    #             self.seen_activities_widgets[activity_id_hash] = set()

    #         if prev_elm_id_hash not in self.seen_activities_widgets[activity_id_hash]:
    #             state_reward += REWARDS["new_state"]
    #         else:
    #             state_reward += REWARDS["repeated_state"]

    #         self.seen_activities_widgets[activity_id_hash].add(prev_elm_id_hash)

    #         # Transition reward
    #         trans_reward = 0.0
    #         trans_key = (prev_activity_id_hash, activity_id_hash)
    #         trans_value = (prev_activity_id_hash, prev_elm_id_hash, activity_id_hash)
    #         if trans_key not in self.seen_transitions:
    #             self.seen_transitions[trans_key] = set()

    #         if trans_value not in self.seen_transitions[trans_key]:
    #             trans_reward += REWARDS["new_transition"]
    #             self.seen_transitions[trans_key].add(trans_value)
    #         else:
    #             trans_reward += REWARDS["repeated_state"]

    #         # Time-scaled reward
    #         time_reward =  1#time_elapsed/ self.max_time if time_elapsed < self.max_time else 0.5

    #         # Coverage delta reward
    #         cov_reward = 0.0
    #         if self.instr_cov == 0.0:
    #             self.instr_cov = instr_cov
    #         else:
    #             cov_reward = (instr_cov - self.instr_cov) * 10
    #             self.instr_cov = instr_cov
            
    #         if self.activity_cov == 0.0:
    #             self.activity_cov = activity_cov
    #         else:
    #             cov_reward += (activity_cov - self.activity_cov) * 10
    #             self.activity_cov = activity_cov

    #         # Total reward
    #         reward += (base_reward + state_reward + trans_reward * self.transition_weight + cov_reward) * time_reward

    #         self.logger.info(
    #             f"Total reward: {reward:.2f}\n==base: {base_reward:.2f},\n==state: {state_reward:.2f}, "
    #             f"\n==trans: {trans_reward* self.transition_weight:.2f},\n==cov: {cov_reward:.2f},\n==time: {time_reward:.2f})"
    #         )

    #         return reward

    #     except Exception as e:
    #         self.logger.error(f"Error calculating reward: {e}")
    #         self.logger.error(traceback.format_exc())
    #         return 0.0
    def calculate_reward(
        self,
        time_elapsed: float,
        found_activities,
        instr_cov: float,
        activity_cov: float,
        prev_activity_id_hash: str,
        prev_elm_id_hash: str,
        activity_id_hash: str,
        state_vector: torch.Tensor,
        has_crash_or_error: bool,
        crash_or_error_logs,
        app_left: bool,
        action_failed: bool
    ) -> float:
        try:
            # === Step penalty ===
            reward = 0.0
            if action_failed:
                return REWARDS["failed_action"]
            
            if app_left and not has_crash_or_error:
                self.logger.warning("App left detected")
                return REWARDS["app_left"]

            # === Crash/Error reward ===
            if has_crash_or_error:
                new_crash_or_error = False
                for crash_or_error_log in crash_or_error_logs:
                    csh = hashlib.md5(crash_or_error_log.strip().encode()).hexdigest()
                    if csh not in self.seen_crashes:
                        new_crash_or_error = True
                        self.seen_crashes.add(csh)
                        self.raw_crashes.append(crash_or_error_log)
                reward += REWARDS["crash_or_err"] if new_crash_or_error else REWARDS["repeated_crash_or_err"]
                self.logger.info(f"{'New' if new_crash_or_error else 'Repeated'}, reward += {reward}")

            # === New activity bonus ===
            new_activity = found_activities.difference(self.seen_activities)
            if new_activity:
                reward += REWARDS["new_activity"]
                self.logger.info(f"New activity {new_activity} found: reward += {REWARDS['new_activity']}")
            self.seen_activities |= found_activities

            # === State similarity reward ===
            state_vector = state_vector.cpu().numpy()
            key = f"{activity_id_hash}_{prev_elm_id_hash}"
            similarity_bonus = 0.0

            if key not in self.seen_states:
                self.seen_states[key] = [state_vector]
                reward += REWARDS["new_state"]
                self.logger.info(f"New state added: {key}")
            else:
                max_sim = 0.0
                for seen_vector in self.seen_states[key]:
                    try:
                        u = state_vector.flatten()
                        v = seen_vector.flatten()
                        if np.linalg.norm(u) == 0 or np.linalg.norm(v) == 0:
                            sim = 1.0
                        else:
                            sim = 1 - cosine(u, v)
                        sim = np.clip(sim, 0.0, 1.0)
                        max_sim = max(max_sim, sim)
                    except Exception as e:
                        self.logger.warning(f"Similarity error: {e}")
                        continue

                if max_sim < self.similarity_threshold:
                    reward += REWARDS["new_state"]
                    self.seen_states[key].append(state_vector)
                elif max_sim > self.familiar_threshold:
                    reward += REWARDS["repeated_state"]
                else:
                    scaled_bonus = (1 - max_sim) * REWARDS["new_state"]
                    reward += scaled_bonus
                    similarity_bonus = scaled_bonus

                self.logger.info(f"State similarity: {max_sim:.2f}, similarity bonus: {similarity_bonus:.2f}")

            # === Widget (state-action) exploration bonus ===
            if activity_id_hash not in self.seen_activities_widgets:
                self.seen_activities_widgets[activity_id_hash] = set()

            if prev_elm_id_hash not in self.seen_activities_widgets[activity_id_hash]:
                reward += REWARDS["new_state"]
            else:
                reward += REWARDS["repeated_state"]

            self.seen_activities_widgets[activity_id_hash].add(prev_elm_id_hash)

            # === Transition reward ===
            trans_reward = 0.0
            trans_key = (prev_activity_id_hash, activity_id_hash)
            trans_value = (prev_activity_id_hash, prev_elm_id_hash, activity_id_hash)
            if trans_key not in self.seen_transitions:
                self.seen_transitions[trans_key] = set()

            if trans_value not in self.seen_transitions[trans_key]:
                trans_reward = REWARDS["new_transition"]
                self.seen_transitions[trans_key].add(trans_value)
            else:
                trans_reward = REWARDS["repeated_state"]
            reward += trans_reward * self.transition_weight

            # === Coverage reward ===
            cov_reward = 0.0
            if self.instr_cov == 0.0:
                self.instr_cov = instr_cov
            else:
                cov_delta = instr_cov - self.instr_cov
                cov_reward += cov_delta * REWARDS["cov"]
                self.instr_cov = instr_cov
            
            if self.activity_cov == 0.0:
                self.activity_cov = activity_cov
            else:
                act_delta = activity_cov - self.activity_cov
                cov_reward += act_delta * REWARDS["cov"]
                self.activity_cov = activity_cov

            reward += cov_reward

            # === Time scaling (optional) ===
            # reward *= (time_elapsed / self.max_time) if time_elapsed < self.max_time else 1.0

            self.logger.info(
                f"Total reward: {reward:.2f} "
                f"(+crash, +activity, +similarity: {similarity_bonus:.2f}, +transition: {trans_reward * self.transition_weight:.2f}, +coverage: {cov_reward:.2f})"
            )

            return reward

        except Exception as e:
            self.logger.error(f"Reward calc error: {e}")
            self.logger.error(traceback.format_exc())
            return 0.0
