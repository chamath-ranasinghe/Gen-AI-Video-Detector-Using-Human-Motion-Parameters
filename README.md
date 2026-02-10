**Project**

SMPL-X expose — utilities and models for deepfake / SMPL-X research and inference.

**Prerequisites**

- Python 3.8+ (a 3.8 venv is included: expose-3.8). Use the system Python that matches your environment.
- Git (for cloning and interacting with the repo).
- Optional: GPU + CUDA drivers for faster inference and training (install appropriate PyTorch wheel).

**Quick Setup (Windows — step-by-step)**

1. Clone the repo (if not already):

**Project**

SMPL-X expose — utilities and models for deepfake / SMPL-X research and inference.

**Prerequisites**

- Python 3.8+ (a 3.8 venv is included: expose-3.8). Use the system Python that matches your environment.
- Git (for cloning and interacting with the repo).
- Optional: GPU + CUDA drivers for faster inference and training (install appropriate PyTorch wheel).

**Setup (Windows)**

1. Clone the repo (if not already):

```powershell
git clone <repo-url>
cd expose
```

2. Create or use a virtual environment

- Use the included venv (if present):

```powershell
# activate included venv
& "expose-3.8\Scripts\Activate.ps1"
```

- Or create a fresh venv and activate it:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Upgrade pip and install dependencies

```powershell
python -m pip install --upgrade pip
# If you need GPU-enabled PyTorch, install it first following https://pytorch.org
pip install -r requirements.txt
```

4. Configure data and checkpoints

- Place model weights and other large artifacts under `data/checkpoints/` or the path your scripts expect.
- Note: this repository's `.gitignore` already includes `data/checkpoints/model.ckpt` to avoid committing large files.

5. Run an example

```powershell
# run demo (if available)
python demo.py

# example inference
python deepfake_predictor.py --input path\to\video.mp4 --output out.json
```

Use `python deepfake_predictor.py --help` to view script-specific flags.

**Setup (Linux / macOS)**

1. Clone and enter the repo:

```bash
git clone <repo-url>
cd expose
```

2. Create and activate a venv:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Upgrade pip and install dependencies:

```bash
python -m pip install --upgrade pip
# If you need GPU-enabled PyTorch, choose the correct install command at https://pytorch.org
pip install -r requirements.txt
```

4. Configure data and run examples:

```bash
# place weights in data/checkpoints/
python demo.py
python deepfake_predictor.py --input path/to/video.mp4 --output out.json
```

**Notes on PyTorch & CUDA**

- If you have an NVIDIA GPU and want GPU acceleration, visit https://pytorch.org and select the correct CUDA-enabled wheel/command for your OS and CUDA driver version. Install that first, then install the remaining requirements.

5. If you run into issues with installing PyTorch try installing it through the wheels by searching at https://pytorch.org/get-started/previous-versions/