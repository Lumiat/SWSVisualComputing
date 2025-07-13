# Facial-Expression-Classifier

> This is an introduction on how to run Facial-Expression-Classifier properly.

## Create Environment

Before you start, you need to set up a conda environment.

### 1. Create Conda Environment

```bash
conda create -n visual-computing python=3.12
# conda create -n <your-environment-name> python=3.12
conda activate visual-computing
# conda activate <your-environment-name>
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
# you should install pytorch version according to your OS, package, language and CUDA version, see: https://pytorch.org/get-started/locally/.
# If you are not sure about your CUDA version, use the following command: nvidia-smi
```

### 2. Install Other Dependencies

```bash
git clone xxx
cd xxx
pip install -r requirements.txt
```

## Quick Start

### 1. Run Real-Time Facial Expression System

run `./application.py` to start the real-time facial expression recognition system:

```bash
python application.py --model raf-db --window_size 5 --mode frequency
# python application.py --model <your-chosen-model> --mode <your-chosen-mode>
```

If you don't know what model or mode is provided, use the following command to help:

```bash
python application.py --help
# see the usage of each argument
```

Click 'Enter' then the system will run, you will see a new window named **Real-time Emotion Recognition** pops out.

### 2. Close Real-Time Facial Expression System

You can stop the system from running by enter 'q' or click the close button of the window.
