import threading
from experiments.testing_controller import TestController

def main():
    # Initialize EnviromentHandler with configuration
    test_controller = TestController(
        emulator_name="avd002",
        appium_port=4723,
        emulator_port=5554,
        apk_path="./apk/de.danoeh.antennapod_3080095.apk",
    )
    # test_controller2 = TestController(
    #     emulator_name="avd002",
    #     appium_port=4723,
    #     emulator_port=5554,
    #     apk_path="./apk/NewPipe_v0.27.7.apk",
    # )
    
    
    # Run testing in a separate thread
    test_controller.run_testing(time_limit=3)
    #test_controller2.run_testing(time_limit=3)


if __name__ == "__main__":
    main()
