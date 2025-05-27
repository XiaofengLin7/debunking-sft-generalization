# REIL
Reasoning Enhanced Imitation Learning

## TODO
- [] long chain of thought rewards
- [] sft scripts

## Experiments roadmap
Diverse dataset, including math dataset, sokoban dataset.
Sokoban dataset should be diverse enough with different length of horizons and different size of maps, etc.
Increase the context windows.
Implement long chain of thought rewards.

## Initial steps on scc
configure user name and email

```bash
git config --global user.name XiaofengLin7
git config --global user.email xfl199801@gmail.com
```

## Installation

```bash
conda create -n reil python=3.9
conda activate reil
git submodule init
git submodule update
pip install -e thirdparty/verl --no-dependencies
pip install -e thirdparty/ragen --no-dependencies
pip install -e thirdparty/alfworld --no-dependencies
pip install -e thirdparty/trl
conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit -y
export CUDA_HOME=$CONDA_PREFIX
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip3 install flash-attn --no-build-isolation
pip install -r requirements.txt
```

## Hardware Requirements

Bfloat16 is only supported on GPUs with compute capability of at least 8.6.
For 3B model, A100 is recommended.
For 1.5B/0.5B model, scripts can be run on L40s.

## update submodules if necessary

```bash
cd thirdparty
cd ragen
git fetch
git checkout branch-name # for ragen
pip install -e . --no-dependencies
cd ..
cd verl
git pull # for verl
pip install -e . --no-dependencies

cd ../..
git add .
git commit -m "update submodule"
git push
```
