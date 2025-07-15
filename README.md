# 🤖 DACN : Automatic black-box testing ultilizing reinforcement learning for Android application

## 📱 Overview

An automated mobile application testing framework combining Deep Q-Learning with GINE for graph feature extraction and CNN for Image patching for efficient test action selection and execution. This model is inspired by the paper [Deep Reinforcement Learning for Automated Android GUI Testing](https://dl.acm.org/doi/10.1145/3597503.3623344)

<p align="center">
  <img src="asset/images/model_wf_gen.png" width="500" alt="Workflow Overview">
</p>

## Member:
- Hồ Vĩnh Nhật - 22521013
- Trần Nguyễn Tiến Thành - 22521364
## 🛠️ Dependencies & Environment

**Core Requirements**:

- Python 3.12.10
- CUDA 12.6
- PyTorch 2.6.0
- Appium 2.17.0 and Appium Python Client 5.0.0
- JDK 17.0.12

**Testing Tools**:

- Android Studio Meerkat 2024.3.1
- Android OS 7.0 (Nougat)
- [ACVTool 2.3.4 Multidex](https://github.com/pilgun/acvtool)

Full dependencies list available in `requirements.txt`

## 📥 Installation

**Setup Environment**:

```bash
git clone https://github.com/Witnull/DACN.git
cd DACN
python -m venv venv
venv/Scripts/activate
pip install -r requirements.txt
```

**Configure Environment Variables**:

```bash
export ANDROID_HOME="/path/to/android/sdk"
export JAVA_HOME="/path/to/jdk-17"
export PATH="$PATH:$ANDROID_HOME/platform-tools"
```

**Tool Installation**:

- Install Android Studio & configure SDK
- [Install Appium](https://appium.io/docs/en/latest/quickstart/install/)
- Run Appium Doctor to verify setup
- Install and setup [ACVTool](https://github.com/pilgun/acvtool)

**Run Test**:
Edit the `main.py` file to configure the test parameters/apks.
Then run to begin the test:

```bash
python ./main.py -t <minutes>
```

## 📁 Project Structure

```
DACN/
├── main.py                     # Main entry point for running tests
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── .gitignore                  # Git ignore rules
│
├── apk/                       # APK collection
│   ├── app-*.apk              # Various application APKs
│   ├── com.github.libretube_62.apk
│   ├── NewPipe_v0.27.7.apk
│   ├── Seal-1.13.1-universal-release.apk
│   ├── instr/                 # Instrumented APKs directory
│
├── asset/                      # Documentation and demo assets
│   ├── images/                # Screenshots and workflow diagrams
│   ├── Demo_our.webm          # Project demonstration video
│   ├── test_model_3mins.csv   # Test results data
│   ├── ComparisonTools.xlsx   # Performance comparison data
│   └── *.pdf                  # Research papers and documentation
│
├── experiments/                # Core framework implementation
│   ├── duel_dqn_agent.py      # Double-Dueling DQN agent implementation
│   ├── gui_embedder.py        # GUI state feature extraction (GNN + CNN)
│   ├── state_embedder.py      # State representation and embedding
│   ├── env_handler.py         # Environment management and interaction
│   ├── testing_controller.py  # Main testing orchestration
│   ├── reward_analyzer.py     # Reward calculation and analysis
│   ├── input_inference.py     # SLM-based input generation
│   ├── logcat_extractor.py    # Android log parsing and analysis
│   ├── logger.py              # Logging utilities
│   └── utils/                 # Helper functions and utilities
│
├── external_tools/             # Third-party tools and dependencies
│   ├── apktool_2.11.1.jar     # APK decompilation tool
│   ├── ACVPatcher-windows/    # Android Code Coverage Patcher
│   └── jdk-17.0.12_windows-x64_bin/ # Java Development Kit
│
├── Logs/                       # Test execution logs and results
│   └── avd003_pfa-notes-*/    # Timestamped test session logs
│
├── venv/                       # Python virtual environment
```

### 📋 Key Components Description

**Core Framework (`experiments/`)**:

- **DQN Agent**: Implements Double-Dueling Deep Q-Network for intelligent action selection
- **GUI Embedder**: Extracts features from Android UI using Graph Neural Networks (GINEConv) for UI hierarchy and CNN for visual elements
- **State Embedder**: Combines GUI features with application state for comprehensive representation
- **Environment Handler**: Manages Android emulator, Appium server, and test execution environment
- **Testing Controller**: Orchestrates the entire testing workflow and coordinates components

**APK Management (`apk/`)**:

- Collection of Android applications for testing
- Instrumented versions for code coverage tracking
- Sample applications from research datasets (AndroTest24, F-Droid)

**External Tools (`external_tools/`)**:

- **ACVTool**: Android Code Coverage instrumentation


**Results (`Logs/`)**:

- Test execution logs with timestamps
- Coverage reports and performance metrics

## ⚙️ How It Works

The framework operates in three main phases:

**🚀 Initialization**

- APK instrumentation for coverage tracking
- Android emulator & Appium server startup
- Initial state analysis

**🔄 Testing Loop**

- GUI state extraction & vectorization using GNN (GINEConv) for Graph, CNN for Image patches, sentence-transformers/all-MiniLM-L6-v2 for Text features
- Action selection using Double-Dueling DQN
- Test execution via Appium - using SLM to generate test inputs (microsoft/Phi-4-mini-instruct)
- Coverage & reward calculation
- Save to Prioritized Experience Replay buffer (PER)
- Train and update DQN model
- Repeat until stopping criteria met

**📊 Results Analysis**

- Coverage
- Bug list

## 🎥 Demo:

https://github.com/user-attachments/assets/c0444c1c-e285-4fad-9120-c4cc8e203dda

## 📈 Results

The test experiment on dataset from [DQT - AndroTest24](https://github.com/Yuanhong-Lan/AndroTest24) with 5 apps and one app from F-Droid.
Each app is given 2 hrs per test model.

### 📊 Coverage Results:

<p align="center">
  <img src="asset/images/Cov_res.png" width="800" alt="Coverage Results">
</p>

#### 🐛 Bug found by our model:

<p align="center">
  <img src="asset/images/errfound.png" width="800" alt="Bug List">
</p>
<p align="center">
  <img src="asset/images/err_types.png" width="800" alt="Bug Types">
</p>

## 📚 References

- [Appium Documentation](https://appium.io/docs/en/2.0/)
- [ACVTool Repository](https://github.com/pilgun/acvtool)
- [Android Testing Guide](https://developer.android.com/training/testing)
- [PyTorch DQN Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
- [PyTorch GINEConv](https://pytorch-geometric.readthedocs.io/en/2.6.0/generated/torch_geometric.nn.conv.GINEConv.html)
- [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [microsoft/Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct)
- [DQT - Deep Reinforcement Learning for Automated Android GUI Testing](https://dl.acm.org/doi/10.1145/3597503.3623344)
- [UI/Application Exerciser Monkey](https://developer.android.com/studio/test/other-testing-tools/monkey)

---

## 🖼️ Images

### Full workflow

<p align="center">
  <img src="asset/images/model_full_wf.png" width="800" alt="Full workflow">
</p>

### Process APKs

<p align="center">
  <img src="asset/images/process_apk.png" width="300" alt="Process APKs">
</p>

### GUI Embedder

<p align="center">
  <img src="asset/images/guiem_wf.png" width="500" alt="GUI Embedding">
</p>

### State Embedder

<p align="center">
  <img src="asset/images/stateem_wf.png" width="300" alt="State Embedding">
</p>

### DQN Model

<p align="center">
  <img src="asset/images/dqna_wf.png" width="300" alt="DQN Model">
</p>

### Environment

<p align="center">
  <img src="asset/images/env_wf.png" width="300" alt="Environment">
</p>
