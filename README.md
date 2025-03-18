# REIL
Reasoning Enhanced Imitation Learning

## TODO
- [create dataset]

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
cd verl
pip install -e . --no-dependencies
cd ..
cd ragen
pip install -e . --no-dependencies
cd ..
conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit -y
export CUDA_HOME=$CONDA_PREFIX
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip3 install flash-attn --no-build-isolation
pip install -r requirements.txt
```

## update submodules if necessary

```bash
cd ragen
git fetch
git checkout branch-name # for ragen
pip install -e . --no-dependencies
cd ..
cd verl
git pull # for verl
pip install -e . --no-dependencies

cd ..
git add .
git commit -m "update submodule"
git push
```
