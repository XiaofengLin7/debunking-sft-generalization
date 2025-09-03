# REIL
We study what makes LLM generalize.

## Prerequisites
CUDA 12.2 & cuDNN 9.1.0 works, but [official docs](https://verl.readthedocs.io/en/latest/start/install.html) recommends CUDA >= 12.4 & cuDNN >= 9.8.0.

## Setup

```bash
conda create -n reil python=3.10
conda activate reil
USE_MEGATRON=0 bash setup.sh
git submodule init
git submodule update
pip install -e thirdparty/verl --no-dependencies
pip install -e thirdparty/ragen --no-dependencies
pip install -e thirdparty/alfworld --no-dependencies
pip install -e thirdparty/trl --no-dependecies
```

## Update submodules if necessary

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
