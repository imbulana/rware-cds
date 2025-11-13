## Setup

Create a conda environment with python 3.11

```bash
conda create -n rware python=3.11
conda activate rware
```

Install the requirements

```bash
pip install -r requirements.txt
```

### Training

See available hyperparams with

```bash
python train_cds.py --help
```

Modify the hyperparams in the [`config_cds.yaml`](config_cds.yaml)` file or set them in the command line

```bash
python train_cds.py \
    --env rware:rware-tiny-2ag-v2 \
    --n_iterations 1000 \
    --n_steps 128 \
    --n_epochs 4 \
    --batch_size 64 \
    --hidden_dim 128 \
    --agent_specific_dim 64 \
    --l1_coef 0.01 \
    --mi_coef 0.1 \
    --device cuda
```