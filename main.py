import threading
from experiments.env_handler import EnviromentHandler
import sys
import signal

env_handler = None


def _sigint_handler(sig, frame):
    print("\nSIGINT detected, stopping gracefully...")
    if env_handler:
        env_handler.cleanup()
    sys.exit(0)


# Register SIGINT handler
signal.signal(signal.SIGINT, _sigint_handler)


def main():
    # Initialize EnviromentHandler with configuration
    env_handler = EnviromentHandler(
        emulator_name="avd002",
        appium_port=4723,
        emulator_port=5554,
        apk_path="./apk/org.totschnig.myexpenses_794.apk",
    )

    # Run testing in a separate thread
    testing_thread = threading.Thread(target=env_handler.run_testing)
    testing_thread.start()
    testing_thread.join()


if __name__ == "__main__":
    main()
