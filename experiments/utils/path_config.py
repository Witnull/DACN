import os

WORK_DIR = "."
APK_DIR = "app"
ANDROID_HOME = "C:/Users/apax/AppData/Local/Android/Sdk"
JAVA_HOME = f"{WORK_DIR}/jdk-17.0.12_windows-x64_bin/jdk-17.0.12"
NODE_PATH = r"C:/Program Files/nodejs/node.exe"
EMULATOR_PATH = f"{ANDROID_HOME}/emulator"
ADB_PATH = f"{ANDROID_HOME}/platform-tools/adb"
APKSIGNER_PATH = f"{ANDROID_HOME}/build-tools/36.0.0/lib/apksigner.jar"

# Set environment variables
os.environ['ANDROID_HOME'] = ANDROID_HOME
os.environ['JAVA_HOME'] = JAVA_HOME
os.environ['PATH'] += (
    f";{EMULATOR_PATH};{os.path.dirname(ADB_PATH)};{JAVA_HOME}/bin;{os.path.dirname(NODE_PATH)}"
)
