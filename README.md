# Debunk the Myth of SFT Generalization
SFT can generalize as well as—or better than—RL when trained with the right data.
## Installation
### Prerequisites
CUDA 12.2 & cuDNN 9.1.0 works, but [official docs](https://verl.readthedocs.io/en/latest/start/install.html) recommends CUDA >= 12.4 & cuDNN >= 9.8.0.

### Setup

```bash
conda create -n debunk_sft python=3.10
conda activate debunk_sft
USE_MEGATRON=0 bash setup.sh
git submodule init
git submodule update
pip install -e thirdparty/verl --no-dependencies
pip install -e thirdparty/ragen --no-dependencies
pip install -e thirdparty/alfworld --no-dependencies
pip install -e thirdparty/trl --no-dependecies
```


## Getting Started
## Dataset
[Dataset collection](https://huggingface.co/collections/Xiaofeng77/debunk-the-myth-of-sft-generalization-68dabd91cad140030b389163)

| Task | Method | Diversity | Format | Link |
| --- | --- | --- | --- | --- |
| Sokoban | RL | non-diverse | — | [🤗](https://huggingface.co/datasets/Xiaofeng77/sokoban) |
| Sokoban | RL | diverse | — | [🤗](https://huggingface.co/datasets/Xiaofeng77/diverse_sokoban) |
| Sokoban | SFT | non-diverse | answer-only | [🤗](https://huggingface.co/datasets/Xiaofeng77/answer-only-sokoban) |
| Sokoban | SFT | diverse | answer-only | [🤗](https://huggingface.co/datasets/Xiaofeng77/diverse-answer-only-sokoban) |
| Sokoban | SFT | non-diverse | cot | [🤗](https://huggingface.co/datasets/Xiaofeng77/cot-sokoban) |
| Sokoban | SFT | diverse | cot | [🤗](https://huggingface.co/datasets/Xiaofeng77/diverse-cot-sokoban) |
| General Points | RL | non-diverse | — | [🤗](https://huggingface.co/datasets/Xiaofeng77/gp-l-only-10k) |
| General Points | RL | diverse | — | [🤗](https://huggingface.co/datasets/Xiaofeng77/diverse-gp-l-only-10k) |
| General Points | SFT | non-diverse | answer-only | [🤗](https://huggingface.co/datasets/Xiaofeng77/answer-only-gp-l-only-10k) |
| General Points | SFT | diverse | answer-only | [🤗](https://huggingface.co/datasets/Xiaofeng77/diverse-answer-only-gp-l-only-10k) |
| General Points | SFT | non-diverse | cot | [🤗](https://huggingface.co/datasets/Xiaofeng77/cot-gp-l-only-10k) |
| General Points | SFT | diverse | cot | [🤗](https://huggingface.co/datasets/Xiaofeng77/diverse-cot-gp-l-only-10k) |
## Train your model with SFT
Specify your model and data beforhand.
For sokoban
```
bash debunk_sft/scripts/sokoban/sokoban_train_and_eval.sh
```
For general points
```
bash debunk_sft/scripts/gp_l/gp_l_train_and_eval.sh
```

## Train your model with RL

Specify your model and data beforhand. For sokoban
```
bash debunk_sft/scripts/sokoban/sokoban_grpo.sh
```
For gp
```
bash debunk_sft/scripts/gp_l/gp_l_grpo.sh
```