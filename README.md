# ProtoWNet - R255 Mini Project

It contains the source code of ProtoWNet, the prototypical weighted network proposed as the [R255 Mini Project](https://www.cl.cam.ac.uk/teaching/2223/R255/) - Topic Explainable AI

## Report 

The report of this project is the [R255_Topic5_gd489.pdf](R255_Topic5_gd489.pdf) file.

## Code

[experiments.py](./experiments/experiments.py) is the main file.
It can be run using W&B sweeps. 

After setting up W&B account on your PC, you can build a config file like [sequential_sweep.yaml](./config/sequential_sweep.yaml).
Then, you set up the sweep:

```
wandb sweep --project NAME_OF_PROJECT ./config/sequential_sweep.yaml
```

It will give you an agent to run

```
wandb agent YOUR_AGENT
```

Remember to change the ```tag``` variable in the [experiments.py](./experiments/experiments.py) file to one of the possible solution (```random```, ```sequential```, ```joint```).

## Setup

```
pip install -r requirements.txt
```
