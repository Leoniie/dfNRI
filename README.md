# Dynamic Factorised Neural Relational Inference
This repository contains my group's research project for the course "Deep Learning" given at ETH Zurich during the autumn semester 2020. The final paper is in [dfNRI.pdf](dfNRI.pdf).

This codebase is largely based on https://github.com/cgraber/cvpr_dNRI. The main changes were made in [dnri.py](dnri/models/dnri.py). Smaller adjustments were made in the following files: [bball_data.py](dnri/datasets/bball_data.py), [bball_experiment.py](dnri/experiments/bball_experiment.py), [motion_experiment.py](dnri/experiments/motion_experiment.py), [model_builder.py](dnri/models/model_builder.py), [evaluate.py](dnri/training/evaluate.py), [flags.py](dnri/utils/flags.py), [misc.py](dnri/utils/misc.py).

## Requirements
The code was written using Python 3.8.5. The requirements are listed in [requirements.txt](requirements.txt).

## Run Experiments
To run this code, you should pip install it in editable mode. This can be done using the following command from the project's root folder: 

```
pip install -e ./
```

In order to run the experiments, you can run the scripts (without the "_leonhard" suffix) found in the [run_scripts](run_scripts/) directory. If you are using ETH Zurich's Leonhard Cluster, you can run the scripts with suffix "_leonhard". Switching from the dNRI model to the dfNRI model requires modifying the following two command line arguments

```
--model_type dfnri --layer_num_edge_types 2 2
```

and deleting the argument `--num_edge_types 4`. Note that `--layer_num_edge_types` is flexible, e.g. the following are possible:
```
--layer_num_edge_types 2 3 4
--layer_num_edge_types 2 3 5 8
```

## Datasets
The motion capture and synthetic datasets are already in the [data](data/) directory.

Download links:
- Motion Capture: the datasets can be downloaded from http://mocap.cs.cmu.edu/search.php?subjectnumber=118 
  and http://mocap.cs.cmu.edu/search.php?subjectnumber=35. For subject 35, you need trials 1-16 and 28-34. For subject 118, you need trials 1-30.
- Stats Perform Basketball Research Datasets: The data can be accessed on [AWS Data Exchange](https://aws.amazon.com/marketplace/pp/prodview-7kigo63d3iln2?qid=1606330770194&sr=0-1&ref_=srh_res_product_title#offers).
- InD: Data must be requested from: https://www.ind-dataset.com/
- Synth: this code includes the synth data, as well as code used to generate it.
