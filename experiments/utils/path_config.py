import os

WORK_DIR = "."
TOOL_DIR = f"{WORK_DIR}/external_tools"
APK_DIR = "apk"
ANDROID_HOME = "C:/Users/apax/AppData/Local/Android/Sdk"
EMULATOR_PATH = f"{ANDROID_HOME}/emulator/emulator.exe"
ADB_PATH = f"{ANDROID_HOME}/platform-tools/adb.exe"
APKSIGNER_PATH = f"{ANDROID_HOME}/build-tools/36.0.0/lib/apksigner.jar"
NODE_PATH = r"C:/Program Files/nodejs/node.exe"

JAVA_HOME = f"{TOOL_DIR}/jdk-17.0.12_windows-x64_bin/jdk-17.0.12"
JAVA_PATH = f"{JAVA_HOME}/bin/java.exe"

# Need ACVTool for code coverage
# notepad 'E:\\DACN\\Model\\DACN\\venv\\Lib\\site-packages\\acvtool\\smiler\\config.json'
# {
#     "AAPT": "C:/users/apax/appdata/local/android/sdk/build-tools/35.0.0/aapt2.exe",
#     "ZIPALIGN": "C:/users/apax/appdata/local/android/sdk/build-tools/35.0.0/zipalign.exe",
#     "ADB": "C:/users/apax/appdata/local/android/sdk/platform-tools/adb.exe",
#     "APKSIGNER": "C:/users/apax/appdata/local/android/sdk/build-tools/35.0.0/apksigner.bat",
#     "ACVPATCHER": "E:\DACN\Model\DACN\external_tools\ACVPatcher-windows\ACVPatcher.exe"
# }


# Set environment variables
os.environ["ANDROID_HOME"] = ANDROID_HOME
os.environ["JAVA_HOME"] = JAVA_HOME
os.environ["PATH"] += (
    f";{EMULATOR_PATH};{os.path.dirname(ADB_PATH)};{JAVA_HOME}/bin;{os.path.dirname(NODE_PATH)}"
)
