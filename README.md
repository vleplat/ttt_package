# Tubal Tensor Train (TTT) — Python package

## Overview

This repository provides a Python implementation of the tubal tensor train (TTT) decomposition.
TTT is a tensor-network model that combines the t-product algebra of T-SVD (with a distinguished tube mode)
and the low-order core structure of the tensor train (TT) format.

For an order-$(N+1)$ tensor with a distinguished tube mode (length $T$), a TTT representation consists of:

- Two boundary tubal cores and $N-2$ interior tubal cores, connected via the t-product
- Storage scaling linearly in the number of modes for bounded tubal ranks (avoids high-order-core growth)

Implemented algorithms:

- **TTT-SVD**: sequential, fixed-rank construction (TT-SVD-style, but with local truncated T-SVD steps)
- **TATCU**: Fourier-slice alternating refinement (ATCU on each frequency slice + rank synchronization)

### Core convention

All TTT cores are stored as 4D arrays with shape:

$
(r_{n-1}, I_n, r_n, T),
\qquad r_0=r_N=1.
$

## Quick start (recommended)

### Step 1: Install Python (3.9+)

If you don't have Python installed:

1. Go to [python.org](https://www.python.org/downloads/)
2. Download **Python 3.9+**
3. Install with default settings
4. Verify in a terminal:

```bash
python3 --version
```

If `python3` is not available on your system, try:

```bash
python --version
```

**Windows (PowerShell)**:

```powershell
py --version
python --version
```

### Step 2: Download this repository

#### Option A (recommended): clone with git

```bash
git clone <REPO_URL_HERE>
cd ttt_package
```

**Windows (PowerShell)**:

```powershell
git clone <REPO_URL_HERE>
cd ttt_package
```

#### Option B: download ZIP

1. Go to the repository page
2. Click **Code** → **Download ZIP**
3. Extract it to a folder
4. Open a terminal in that folder

### Step 3: Create a virtual environment

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

**Windows (PowerShell)**:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
```

**Windows (cmd.exe)**:

```bat
py -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -U pip
```

### Step 4: Install `ttt_package`

```bash
python -m pip install -e .
```

**Windows (PowerShell)**:

```powershell
python -m pip install -e .
```

### Step 5 (optional): Install legacy dependencies

Some legacy audited backends depend on `tensorly` and `scipy`:

```bash
python -m pip install -e ".[legacy]"
```

## Running the demo (chromatic gesture video)

The main demo script is:

- `test/demo_ttt_tatcu_chromatic_gesture.py`

It generates a synthetic RGB video, reshapes it into a higher-order tensor (keeping time as the tube mode),
then compares **TTT-SVD** and **TATCU** reconstructions.

Run (default = fixed-rank TATCU):

```bash
python test/demo_ttt_tatcu_chromatic_gesture.py
```

or (module form):

```bash
python -m test.demo_ttt_tatcu_chromatic_gesture
```

### Running TATCU with different settings

The demo exposes the three TATCU variants through `--tatcu_mode`:

#### 1) Fixed-rank TATCU (baseline / current default)

```bash
python test/demo_ttt_tatcu_chromatic_gesture.py \
  --tatcu_mode fixed_rank \
  --tt_ranks 1 4 6 6 3 1 \
  --max_sweeps 4
```

- `--tt_ranks`: **full** TT rank profile (including boundary 1s), used on each Fourier slice.
- `--max_sweeps`: number of left-right/right-left ATCU sweeps per processed slice.

#### 2) Slice-adaptive TATCU (MATLAB-faithful slice budget)

```bash
python test/demo_ttt_tatcu_chromatic_gesture.py \
  --tatcu_mode slice_adaptive \
  --slice_rel_tol 0.11 \
  --tt_ranks 1 4 6 6 3 1 \
  --max_tt_ranks 1 8 10 10 6 1 \
  --max_sweeps 4
```

- `--slice_rel_tol`: per-slice **relative** target used to form a slice error budget in the Fourier domain.
- `--tt_ranks`: initial TT rank profile.
- `--max_tt_ranks`: maximum rank caps for per-slice rank growth (if the target cannot be met at the initial ranks).

#### 3) Global-tolerance TATCU (Parseval-budgeted slices)

```bash
python test/demo_ttt_tatcu_chromatic_gesture.py \
  --tatcu_mode global_tol \
  --eps_rel 0.11 \
  --tt_ranks 1 4 6 6 3 1 \
  --max_tt_ranks 1 8 10 10 6 1 \
  --max_sweeps 2
```

- `--eps_rel`: **global** target for \(\|X-\hat X\|_F/\|X\|_F\) (the code verifies this when possible).
- `--tt_ranks`: initial rank profile.
- `--max_tt_ranks`: maximum rank caps for slice-wise rank growth.

**Important:** a global tolerance does **not** imply a per-frame guarantee; the demo reports per-frame errors as diagnostics.

#### Optional: overlay a target line on the per-frame error plot

```bash
python test/demo_ttt_tatcu_chromatic_gesture.py \
  --tatcu_mode global_tol --eps_rel 0.11 \
  --frame_target 0.11
```

This only affects the plot (a dashed horizontal line); it is not enforced per frame.

**Note (macOS / Matplotlib):** this demo forces Matplotlib’s non-interactive backend (`Agg`) to avoid
native GUI backend crashes. It only saves figures; it does not open any windows.

### Outputs

Figures are saved under `output_figures/` as **both PNG and PDF**:

- `output_figures/gesture_comparison__<tag>.(png|pdf)`
- `output_figures/gesture_per_frame_errors__<tag>.(png|pdf)`

(A small comparison GIF is also produced: `output_figures/gesture_comparison__<tag>.gif`.)

## TATCU API notes (important)

This repo currently provides **three** TATCU entry points:

- `tatcu_fixed_rank`: fixed TT-rank profile per Fourier slice (this is the current default)
- `tatcu_slice_adaptive`: slice-wise adaptive ranks using a slice error budget, then rank synchronization
- `tatcu_global_tol`: wrapper that allocates slice budgets from a **global** relative target \(\varepsilon\)

For backward compatibility:

- `tatcu = tatcu_fixed_rank`
- `tatcu_prototype = tatcu_fixed_rank`

## Project structure

```
ttt_package/
  src/ttt_package/
    __init__.py
    core.py
    tproduct.py
    tsvd.py
    ttt_svd.py
    tatcu.py
    tt_backend.py
    legacy/
      __init__.py
      tt_lib.py
      tucker2_lib.py
  legacy_backends/
    tt_lib.py
    tucker2_lib.py
  test/
    demo_ttt_tatcu_chromatic_gesture.py
  tests/
    test_tatcu.py
  output_figures/
  pyproject.toml
  README.md
```

## References



## 📖 How to Cite

If you use this code in academic work, please cite the accompanying paper draft 

```
@misc{ahmadiasl2026newtensornetworktubal,
      title={A New Tensor Network: Tubal Tensor Train and Its Applications}, 
      author={Salman Ahmadi-Asl and Valentin Leplat and Anh-Huy Phan and Andrzej Cichocki},
      year={2026},
      eprint={2603.10503},
      archivePrefix={arXiv},
      primaryClass={math.NA},
      url={https://arxiv.org/abs/2603.10503}, 
}
```

and include a reference to this repository.

## 📧 Support and Contact

For questions, bug reports, or contributions, please contact:
**valentin dot leplat [at] gmail dot com**

