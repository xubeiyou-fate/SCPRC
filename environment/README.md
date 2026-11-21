# Environment

Checkpoint inference was checked with Python 3.12.7, PyTorch 2.7.1+cu118, NumPy 1.26.4, and pandas 2.2.2. The recorded environment is in [environment metadata](migration_environment.json). Individual training environments are recorded with their experiments.

## Installation

Create a Python 3.12 environment and install the PyTorch build for your hardware:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# CUDA 11.8
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
# For CPU inference, use this command instead of the CUDA line:
# python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

The adapter includes the limited Lightning compatibility interface needed by the supplied inference code. It does not provide the upstream Lightning training loop. Use the repository requirements and experiment-specific entry points rather than combining the requirements of all upstream projects.

GPU outputs can vary across devices and library builds. Recorded parameter hashes and fixed-input checks describe checkpoint consistency; uncertainty in forecasting metrics is reported separately in the experiment results.
