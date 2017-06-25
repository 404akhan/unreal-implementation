# unreal

UNREAL paper https://arxiv.org/pdf/1611.05397.pdf

Implementation of three Auxiliary Tasks: Pixel Cotrol, Value Function Replay, Reward Prediction built on top of advantage actor-critic

In plots we see that it gives some speed up on learning easy DoomBasic-v0 environment

To see plots with tensorboard run: tensorboard --logdir=logs

![Alt text](plots/length.png?raw=true "Episode Length")
![Alt text](plots/reward.png?raw=true "Episode Reward")
![Alt text](plots/runs.png?raw=true "Different Runs")