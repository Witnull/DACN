import argparse
import signal
import sys
import time
from experiments.input_inference import InputSuggestionLLM
from experiments.testing_controller import TestController


def handle_sigint(signum, frame):
    global interrupted
    print("\n[INFO] SIGINT received. Gracefully shutting down...")
    sys.exit(0)


signal.signal(signal.SIGINT, handle_sigint)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run GUI testing with RL agent")
    parser.add_argument(
        "-t",
        "--time-limit",
        type=int,
        default=3,
        help="Time limit for testing in minutes (default: 3)",
    )
    parser.add_argument(
        "-e",
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to run (default: 5). if set time_limit then this is ignored",
    )
    parser.add_argument(
        "-s",
        "--steps",
        type=int,
        default=16 * 5,
        help=f"Maximum steps per episode (default: {16 * 5})",
    )

    args = parser.parse_args()
    apks = [
        "./apk/pfa-notes-release-v1.4.1.apk",
        "./apk/com.tombursch.kitchenowl_115.apk",
        # "./apk/NewPipe_v0.27.7.apk",
        # "./apk/Seal-1.13.1-universal-release.apk",
        # "./apk/app-release.apk",
        #"./apk/com.github.libretube_62.apk",
        #"./apk/com.shabinder.spotiflyer_32.apk",
        #"./apk/MyExpenses-r554-debug.apk",
    ]
    # Initialize EnviromentHandler with configuration
    txt_generator = InputSuggestionLLM(model_name="microsoft/Phi-4-mini-instruct")
    for i, apk in enumerate(apks):
        test_controller = TestController(
            emulator_name=f"avd00{i + 3}",
            appium_port=4723 + i * 2,
            emulator_port=5554 + i * 2,
            apk_path=apk,
            txt_generator=txt_generator,
        )

        test_controller.run_testing(
            time_limit=args.time_limit, episodes=args.episodes, max_steps=args.steps
        )
        print("[INFO] Testing completed. Cooling down...")
        time.sleep(60)  # long sleep to ensure the app is stable before next test
    print("[INFO] All testing completed.")


if __name__ == "__main__":
    main()
