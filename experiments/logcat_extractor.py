import subprocess
import os

import re
from datetime import datetime
from experiments.logger import setup_logger


class LogcatExtractor:
    def __init__(self, adb_path, device_udid, app_name, logdir):
        self.adb_path = adb_path
        self.device_udid = device_udid
        self.app_name = app_name

        self.logger = setup_logger(
            f"{logdir}/logcat_extractor.log",
            emulator_name=self.device_udid,
            app_name=self.app_name,
        )
        self.logger.info(
            f"Initializing LogcatExtractor for {self.app_name} on {self.device_udid}"
        )

        self.logcat_logdir = f"{logdir}/logcat"
        os.makedirs(self.logcat_logdir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_filename = (
            f"logcat_{self.device_udid}_{self.app_name}_{self.timestamp}"
        )

        self.crash_file = os.path.join(
            self.logcat_logdir, f"CRASH_ERROR_{self.base_filename}.log"
        )
        self.file_index = 0
        self.snapshot_file = os.path.join(
            self.logcat_logdir, f"SNAP_{self.file_index}_{self.base_filename}.log"
        )
        self.coverage_file = os.path.join(
            self.logcat_logdir, f"ACV_COV_{self.base_filename}_coverage.log"
        )

    def clear_logcat(self):
        cmd = [
            self.adb_path,
            "-s",
            self.device_udid,
            "logcat",
            "-c",
        ]
        self.logger.info(
            f"Clearing logcat buffers for {self.app_name} on {self.device_udid}"
        )
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0
    
    def dump_logcat(self):
        cmd = [
            self.adb_path,
            "-s", self.device_udid,
            "logcat",
            "-b", "main",
            "-b", "crash",
            "-d"
        ]

        raw_result = subprocess.run(cmd, capture_output=True, text=True)
        result = "\n".join(line for line in raw_result.stdout.splitlines() if "appium" not in line.lower())
        self.logger.info(
            f"Dumped logcat for {self.app_name} on {self.device_udid}, size: {len(result)} bytes"
        )
        # Save the log to a file
        self.save_log(result)
        return result.strip()

    def extract_crash_logs(self, log):
        crash_patterns = [
            "FATAL EXCEPTION",
            "ANR in",
            "RuntimeException",
            "SIGSEGV",
            "Force finishing activity",
            "has died",
            "native crash",
            "EXCEPTION_ACCESS_VIOLATION",
            "Exception",
            "Error",
        ]
        crash_logs = []
        for line in log.splitlines():
            if any(
                re.search(pattern.lower(), line.lower()) for pattern in crash_patterns
            ):
                line = ' '.join(line.split(":")[-2:]).strip()  # Remove timestamp and tag
                crash_logs.append(line)
        self.save_crash_log(crash_logs)
        return crash_logs

    def extract_acv_coverage(self, log):
        acv_pattern = re.compile(r"covered (\d+) out of (\d+)")
        coverage_lines = []
        total_covered = 0
        total_possible = 0

        for line in log.splitlines():
            if "ACV" in line and "covered" in line:
                match = acv_pattern.search(line)
                if match:
                    covered = int(match.group(1))
                    possible = int(match.group(2))
                    percent = (covered / possible * 100) if possible else 0.0
                    coverage_lines.append(f"{line}  -->  {percent:.2f}%")
                    total_covered += covered
                    total_possible += possible

        total_percent = (
            (total_covered / total_possible) if total_possible else 0.0
        )
        coverage_lines.append(
            f"\nTotal coverage: {total_covered} / {total_possible} --> {total_percent*100:.2f}%"
        )
        self.save_coverage_log(coverage_lines)
        return total_percent

    def contains_crash(self, crash_logs=[]):
        return len(crash_logs) > 0

    def save_log(self, log):
        while (
            os.path.exists(self.snapshot_file)
            and os.path.getsize(self.snapshot_file) >= 50 * 1024 * 1024  # > 50MB
        ):
            self.file_index += 1
            self.snapshot_file = os.path.join(
                self.logcat_logdir, f"SNAP_{self.file_index}_{self.base_filename}.log"
            )

        with open(self.snapshot_file, "a", encoding="utf-8") as f:
            f.write("\n======BEGIN-SNAP=======\n" + log + "\n======EO-SNAP=======\n")

    def save_crash_log(self, crash_logs):
        if crash_logs and len(crash_logs) > 0:
            with open(self.crash_file, "a", encoding="utf-8") as f:
                f.write("\n======= CRASH/ERROR DETECTED =======\n")
                f.write("\n*******###*******\n".join(crash_logs) + "\n")
                f.write("====================================\n")

    def save_coverage_log(self, coverage_lines):
        if coverage_lines and len(coverage_lines) > 0:
            with open(self.coverage_file, "a", encoding="utf-8") as f:
                f.write("\n======= ACV COVERAGE REPORT =======\n")
                f.write("\n*******###*******\n".join(coverage_lines) + "\n")
                f.write("===================================\n")

    # HOW IT WORKS:
    # EACH STEP:
    # 2. Dump logcat output
    # 3. Extract crash logs
    # 4. Extract coverage lines
    # 5. Save logs to files
    # 1. Clear logcat buffers

    # def run(self):
    #     self.clear_logcat()
    #     time.sleep(1)  # Allow system to stabilize
    #     log_output = self.dump_logcat()
    #     crash_logs = self.extract_crash_logs(log_output)
    #     coverage_percent = self.extract_acv_coverage(log_output)
