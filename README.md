# BTC-GPU Quick Start

BTC-GPU is a CUDA-based program for third-order sparse tensor decomposition. It uses adaptive blocking and parallel SGD to learn three factor matrices on one or multiple GPUs.

## Requirements

- Linux or WSL
- NVIDIA GPU and CUDA Toolkit
- `g++` and GNU Make
- Python 3 (optional, for data preparation and batch experiments)

The default CUDA path is `/usr/local/cuda-12.1`, and the target GPU architecture is `sm_70`. Update these values in `Makefile` if your environment differs.

## Build

```bash
make clean
make
```

The compiled executable is `tensor_sgd`.

## Data Format

Prepare the following two files:

```text
data/<dataset>.train
data/<dataset>.test
```

Each line must use the following format:

```text
x y z value
```

Example:

```text
1 1 1 0.82
1 3 2 0.15
2 1 4 0.63
```

## Run

```bash
./tensor_sgd <dataset> <I> <J> <K> <data_dir> <coord_base> <output_dir> [options]
```

Example:

```bash
./tensor_sgd demo 1000 2000 24 ./data 1 ./output \
  --epochs 200 \
  --rank 16 \
  --lr 0.001 \
  --reg 0.05 \
  --block 1024 \
  --thread-size 512 \
  --gpus 1
```

Common options:

| Option | Description |
| --- | --- |
| `--epochs` | Number of training epochs |
| `--rank` | Tensor decomposition rank, from 1 to 100 |
| `--lr` | Learning rate |
| `--reg` | Regularization coefficient |
| `--block` | Target block size |
| `--thread-size` | Number of threads per CUDA block |
| `--gpus` | Number of GPUs to use; `0` uses all visible GPUs |

## Output

Results are written to:

```text
output/<dataset>_<train_rate>/
```

Main output files:

- `error.txt`: RMSE, MAE, and relative error for each epoch
- `time.txt`: total time per epoch
- `kernel_time.txt`: CUDA kernel time
- `sync_time.txt`: multi-GPU synchronization time
- `block_parallel.txt`: block and parallel scheduling information

## Utility Scripts

```text
scripts/prepare_lanl2_3d.py      Prepare a third-order tensor
scripts/prepare_lbnl_3d.py       Convert the LBNL tensor from 5D to 3D
scripts/run_param_sweep.py       Run parameter sweeps
scripts/run_three_datasets.py    Run multiple datasets in a batch
scripts/uniform_block_counts.py  Analyze uniform block distributions
```

See [README.md](README.md) for the complete documentation.
