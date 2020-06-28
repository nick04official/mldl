# First person action recognition

![Python 3.7](https://img.shields.io/badge/Python-3.7-blue.svg?&logo=python&style=flat) ![Pytorch 1.5](https://img.shields.io/badge/Pytorch-1.5-blue.svg?&logo=pytorch&style=flat)

_Deep learning model for action recognition in first person POV_

This project was developed by Nicolò Bertozzi and Francesco Bianco Morghet for the _Machine Learning and Deep Learning_ course at Politecnico di Torino.

## General description

![Network structure](paper/schemi/two_stream2_img.png)

The aim of this project is to perform first person action recognition by leveraging both RGB frames and warped optical flow frames.

Our method consists in adapting the RGB network proposed in [1] to warped optical flow frames: to perform this operation, we followed a procedure very similar to the one proposed in [2]. The proposed network takes both RGB frames and warped optical flow frames as input data.

## Run this project

### System Requirements

 - Ubuntu 18.04, follow [this gist](https://gist.github.com/francibm97/da7a299d40aa7907175e585fc0182d6f) to install the same python environment used in this project
 - CUDA enabled GPU with at least 16GB of VRAM
 - At least 48GB of RAM

Download the GTEA-61 dataset and this project to the same directory. The directory tree should look something like this:

```
├── GTEA61
│   ├── flow_x_processed
│   │   └── ...
│   ├── flow_y_processed
│   │   └── ...
│   └── processed_frames2
│       └── ...
└── mldl
    ├── README.md
    ├── params
    │   └── ...
    └── src
        ├── run.py
        └── ...
```

### Usage

To replicate the 2-stage training procedure of our model, you should
1. Train WFCNet
2. Train the model

To do so:

1. Perform the first stage training of WFCNet: 
```
python3.7 mldl/src/run.py mldl/params/wfcnet_stacked_stage1
```

2. Replace the `_model_state_dict` property of `wfcnet_stacked_stage2` by entering the output path of the previously trained model, then run:
```
python3.7 mldl/src/run.py mldl/params/wfcnet_stacked_stage2
```

3. Replace the `_model_state_dict` property of `wfcnetambi_stacked_14_stage1` by entering the output path of the trained WFCNet module, then run:
```
python3.7 mldl/src/run.py mldl/params/wfcnetambi_stacked_14_stage1
```

4. Replace the `_model_state_dict` property of `wfcnetambi_stacked_14_stage2` to match the output path of the previously trained model, then run:
```
python3.7 mldl/src/run.py mldl/params/wfcnetambi_stacked_14_stage2
```

# References

 - [1] [Attention is all we need: Nailing down object-centric attention for egocentric activity recognition](https://arxiv.org/abs/1807.11794) Swathikiran Sudhakaran and Oswald Lanz, 2018. [[official code]](https://github.com/swathikirans/ego-rnn)
 - [2] [(DE)²CO: Deep depth colorization](https://arxiv.org/abs/1703.10881) F. M. Carlucci, P. Russo, and B. Caputo, 2017.
 - [3] [Joint encoding of appearance and motion features with self-supervision for first person action recognition](https://arxiv.org/abs/2002.03982) Mirco Planamente, Andrea Bottino, and Barbara Caputo, 2020.
 - [4] [Action recognition with improved trajectories](https://hal.inria.fr/hal-00873267v2/document) Heng Wang, 2015.