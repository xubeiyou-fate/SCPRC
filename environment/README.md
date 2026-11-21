# Runtime environment

The migration validation uses Python 3.12.7, PyTorch 2.7.1+cu118, NumPy 1.26.4 and pandas 2.2.2. The complete relevant snapshot is in `migration_environment.json`. This describes the migration checks, not every historical training run.

Create a fresh Python 3.12 environment. Install the desired PyTorch build first:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# CUDA 11.8 build used by the migration checks:
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
# Alternatively, for CPU inference replace the previous line with:
# python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

The forecasting adapter supplies a narrow Lightning compatibility shim when Lightning is absent. It does not implement the upstream Lightning training loop. Do not combine all historical vendor requirements: their old PyTorch constraints conflict with the validated environment. Follow the migrated experiment entry points for the SCPRC protocol.

GPU execution can vary across devices and library builds. Exact checkpoint tensor hashes and fixed-input forward parity are recorded separately from historical metric uncertainty.
