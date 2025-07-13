import subprocess
import sys
import time
import os
from appium import webdriver
from appium.webdriver.appium_service import AppiumService
from appium.options.android import UiAutomator2Options
from selenium.common.exceptions import WebDriverException
from experiments.logger import setup_logger
from .path_config import ADB_PATH, EMULATOR_PATH, APK_DIR, NODE_PATH

appium_apks = [
    f"{APK_DIR}/settings_apk-debug.apk",
    f"{APK_DIR}/appium-uiautomator2-server-v7.4.1.apk",
]


class EmulatorController:
    def __init__(
        self,
        emulator_name: str,
        emulator_port: str,
        log_dir: str = "app/logs/",
        app_name: str = "Unknown",
    ):
        self.emulator_name = emulator_name
        self.emulator_port = emulator_port
        self.logger = setup_logger(
            f"{log_dir}/emu_controller.log",
            emulator_name=self.emulator_name,
            app_name=app_name,
        )
        self.adb_path = ADB_PATH
        self.emulator_exe = EMULATOR_PATH
        self.process = None

    def start_emulator(self) -> bool:
        try:
            self.logger.info(f"Starting emulator: {self.emulator_name}...")
            cmd = (
                f"{self.emulator_exe} -port {self.emulator_port} -avd {self.emulator_name}"
                " -no-snapshot-save -wipe-data -no-boot-anim -accel auto"
            )
            self.process = subprocess.Popen(cmd, shell=True)
            self.logger.info(f"Started emulator: {self.emulator_name}")
            time.sleep(45)
            # disable animations
            for scale in [
                "window_animation_scale",
                "transition_animation_scale",
                "animator_duration_scale",
            ]:
                os.system(
                    f"{self.adb_path} -s emulator-{self.emulator_port} shell settings put global {scale} 0"
                )

            # os.system(
            #     f"{self.adb_path} -s emulator-{self.emulator_port} shell settings put system pointer_location 1"
            # )
            # os.system(
            #     f"{self.adb_path} -s emulator-{self.emulator_port} shell settings put system show_touches 1"
            # )
            # adb shell settings put system show_touches 1
            return True
        except Exception as e:
            self.logger.error(f"Error starting emulator: {e}")
            return False

    def get_device_name(self) -> str:
        attempt = 0
        self.logger.info("Fetching device name...")
        while attempt < 8:
            attempt += 1
            try:
                self.logger.info(f"Attempt {attempt} to get device name...")
                output = (
                    subprocess.check_output([self.adb_path, "devices"])
                    .decode()
                    .splitlines()
                )
                for line in output[1:]:
                    if line.strip() and "device" in line:
                        name = line.split()[0]
                        self.logger.info(f"Device name: {name}")
                        return name
            except Exception as e:
                self.logger.error(f"Error getting device name: {e}")
            time.sleep(2)
        sys.exit(
            "Failed to get device name after multiple attempts. Please check the emulator."
        )

    def install_appium_apks(self) -> bool:
        for i in range(3):  # Retry installation
            self.logger.info(f"Attempt {i + 1} to install Appium APKs...")
            for apk in appium_apks:
                try:
                    cmd = f"{self.adb_path} install -r {apk}"
                    subprocess.run(cmd, shell=True, check=True)
                    self.logger.info(f"Installed APK: {apk}")
                except subprocess.CalledProcessError as e:
                    self.logger.error(f"Failed to install APK {apk}: {str(e)}")
            return True
        raise RuntimeError("Failed to install Appium APKs after multiple attempts.")

    def check_emulator_status(self, device_name: str) -> bool:
        try:
            output = subprocess.check_output(
                f"{self.adb_path} devices", shell=True, text=True
            )
            return f"{device_name}\tdevice" in output
        except Exception as e:
            self.logger.error(f"Error checking status: {e}")
            return False

    def cleanup_emulator(self, device_name: str):
        try:
            cmd = [self.adb_path, "-s", device_name, "emu", "kill"]
            subprocess.run(cmd, shell=True)
            self.logger.info("Emulator cleaned up")
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")


class AppiumManager:
    def __init__(
        self,
        appium_port: int,
        apk_path: str,
        log_dir: str = "app/logs/",
        emulator_name: str = "Unknown",
        app_name: str = "Unknown",
    ):
        self.appium_port = appium_port
        self.apk_path = apk_path
        self.node_path = NODE_PATH
        self.logger = setup_logger(
            f"{log_dir}/appium_manager.log",
            emulator_name=emulator_name,
            app_name=app_name,
        )
        self.appium_service = None
        self.driver = None
        self.device_name = None
        self.exported_activities = []
        self.app_package = None

    def start_appium_server(self) -> bool:
        """Start the Appium server."""
        try:
            self.logger.info(f"Starting Appium server on port {self.appium_port}...")
            self.appium_service = AppiumService()
            self.appium_service.start(
                args=[
                    "--port",
                    str(self.appium_port),
                    "--address",
                    "127.0.0.1",
                ],
                node_path=self.node_path,
            )
            time.sleep(2)  # Wait for Appium to start
            self.logger.info(f"Started Appium service on port {self.appium_port}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to start Appium service: {str(e)}")
            return False

    def connect(self, device_name: str, app_package: str = None) -> bool:
        self.device_name = device_name
        self.logger.info(
            f"Connecting to Appium via device {self.device_name} - {app_package}..."
        )
        # build capabilities
        caps = {
            "platformName": "Android",
            "appium:deviceName": self.device_name,
            "appium:app": self.apk_path,
            "appium:automationName": "UiAutomator2",
            "appium:autoGrantPermissions": True,
            "appium:appWaitActivity": "*",
            "appium:fullReset": False,
            "appium:noReset": False,
            "appium:unicodeKeyboard": True,
            "appium:resetKeyboard": True,
            "appium:androidInstallTimeout": 60000,
            "appium:isHeadless": False,
            "appium:adbExecTimeout": 60000,
            "appium:newCommandTimeout": 60000,
        }
        if app_package:
            caps["appium:appPackage"] = app_package
        options = UiAutomator2Options().load_capabilities(caps)
        for _ in range(3):  # Retry connection
            try:
                self.driver = webdriver.Remote(
                    f"http://127.0.0.1:{self.appium_port}", options=options
                )
                self.logger.info("Connected to Appium driver")
                return True
            except WebDriverException as e:
                self.logger.error(f"Appium connection failed: {e}")
                time.sleep(2)
        return False

    def cleanup_appium(self):
        if self.driver:
            self.driver.quit()
            self.driver = None
            self.logger.info("Appium driver quit")
        if self.appium_service:
            self.appium_service.stop()
            self.appium_service = None
            self.logger.info("Appium service stopped")

    def restart_appium(self):
        """Restart the Appium server and driver."""
        self.logger.info("Restarting Appium service and driver...")
        self.cleanup_appium()
        if not self.start_appium_server():
            self.logger.error("Failed to restart Appium server")
            sys.exit(0)
        if not self.connect(self.device_name, self.app_package):
            self.logger.error("Failed to restart Appium server")
            sys.exit(0)
        self.logger.info("Appium service and driver restarted successfully")
