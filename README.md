# BTC-GPU Quick Start

BTC-GPU is a CUDA-based program for third-order sparse tensor decomposition. It uses adaptive blocking and parallel SGD to learn three factor matrices on one or multiple GPUs.

## Requirements

- Linux or WSL
- NVIDIA GPU and CUDA Toolkit
- `g++` and GNU Make
- Python 3 (optional, for data preparation and batch experiments)
- 
## Build

```bash
make clean
make
```
## Data Format

Prepare the following two files:

```text
data/<dataset>.train
data/<dataset>.test
```

## Run

```bash
./tensor_sgd <dataset> <I> <J> <K> <data_dir> <coord_base> <output_dir> [options]
```


