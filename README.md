# DACN

## Dependencies

- Python 3.12.10
- CUDA 12.6
- torch 2.6.0
- Appium 2.17.0
- Appium-Python-Client 5.0.0
- Android Studio - Meerkat
- Jdk 17

... For more libraries details please read in `req.txt`

## Directory Structure

The current directories of project are as follow, the one that merked as (done) is considered as done and just for reference

```
.
├── app/ (done)
│   ├── logs/
│   │   └── -- training logs for tested apps
│   ├── result/
│   │   └── -- test results (e.g., coverage reports)
│   ├── model/
│   │   ├── model_avd001.pth    -- trained DQN models
│   │   └── -- more models here
│   ├── MyApp.apk
│   └── others.apk
├── experiments/
│   ├── logger.py            # Logger module (done)
│   ├── gui_embedder.py      # GUIEmbedder module
│   ├── action_extractor.py  # ActionExtractor module
│   ├── dqn_agent.py         # DQNAgent module
│   ├── reward_analyzer.py   # RewardAnalyzer module
│   ├── emulator_handler.py  # EmulatorHandler module
│   ├── monteCarloTreeSearch_intergration.py # for MCTS and StateTransition
│   └── uitls/ (done)
│       └── apk_analyzer.py    # analyze apk , taken from ARES 
├── jdk-17 (done)
├── venv (done)
└── main.py                # Entry point, imports from experiments (done)

```
## How to use

1 - Create venv 
2 - Install python libraries
3 - Install Appium, Android studio
4 - Install Appium doctor to check for missing stuff - maybe add ANDROID_HOME and JAVA_HOME to environment vars
5 - Install Jdk
6 - Get some sample apk
7 - Edit the `main.py`
8 - Run the `main.py`
9 - Debug 
