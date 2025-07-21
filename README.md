# REIL
We study what makes LLM generalize.

[Google Doc](https://docs.google.com/document/d/1vZrBTFJIvfnOIr9hGS_SWdRS2Z1xw6kCow7Ps1BWaZc/edit?tab=t.0#heading=h.bsh8cnuvxol2)


## Experiments roadmap
- Comprehensive experiments on SFT vs RL on reasoning-required tasks.
- Study why SFT can outperform RL on Sokoban task: measure $p_r$, $p_g$, $\pi_g$ and $\pi_r$. 

## To dos
- GRPO for 5 domains(Sokoban, sudoku, shortest_path, knights_knaves, countdown)
- SFT for 5 domains
- Score function for SFT(see branch: feature/score_for_sft)
- KL divergence metric for SFT

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
