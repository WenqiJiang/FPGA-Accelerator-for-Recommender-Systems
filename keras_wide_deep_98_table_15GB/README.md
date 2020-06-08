### Keras Implementation of Wide-Deep Model

Same dataset of the tensorflow version.

## Environment

conda create -n py3tf1 python=3.6
conda install pyyaml pandas tensorflow===1.15 keras

## Directories

### cartesian_analysis

Join embeddings (Cartesian product) to reduce random memory access. Analyze the trade-off of storage and memory access.

### conf

Model construction configurations, not all of them are used.

### data

Dataset.

### model

Where the model is stored (in h5 or tf format).

### output

The profiling information in json format. You can type "chrome://tracing/" in your chrome browser and load the json file.

### python

Where the Keras model is constructed.


#### python/logs

The logs for tensorboard. You can run:

```
tensorboard --logdir=python/logs/scalars
```
