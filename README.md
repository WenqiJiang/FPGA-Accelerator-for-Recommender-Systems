# FPGA-Accelerator-for-Recommender-Systems

This is the CPU baseline repository for two high-performance recommendation systems we built, i.e., [MicroRec (MLSys 2021)](https://github.com/fpgasystems/FPGA-Recommendation-Accelerator) and [FleetRec (KDD 2021)](https://github.com/fpgasystems/GPU-FPGA-Recommendation-System).

## Link to the Repository of the wide & deep model (Tensorflow)
https://github.com/Lapis-Hong/wide_deep

Some adjustment are applied to that repo and the new version is stored in "/tf-wide-deep"

## Enviornment setup

### Install Anaconda 

```
wget https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh
chmod +x Anaconda3-2020.07-Linux-x86_64.sh
./Anaconda3-2020.07-Linux-x86_64.sh
```

### Config Anaconda Enviornment

We need 2 different environments, one for training using Tensorflow 1.x while another for serving with Tensoflow 2.x.

training enviornment:

```
conda create -n wide_deep python=2.7 -y
conda activate wide_deep  
conda install -y tensorflow===1.14 pyyaml  
conda install -c qiqiao tensorflow_serving_api -y 
```

serving enviornment: 

```
conda create -n py3tf2 python=3.8 -y
conda activate py3tf2
conda install tensorflow=2.2 -y 
pip install -U tensorboard_plugin_profile
```

### Training

In conf/train.yaml, you can configure whether to use the wide & deep recommendation model or the deep model only.

Training and export model example:

```
cd FPGA-Accelerator-for-Recommender-Systems/tf_wide_deep_47_table/python/
conda activate wide_deep
python train.py –train_epochs 1
python tensorflow_serving/export_savedmodel.py
```

### Serving

Install TF Serving: https://www.tensorflow.org/tfx/serving/setup

```
echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | sudo tee /etc/apt/sources.list.d/tensorflow-serving.list && \
curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | sudo apt-key add -
sudo apt-get update && sudo  apt-get install tensorflow-model-server
```

Start serving:

```
tensorflow_model_server --port=9000 --model_name=wide_deep --model_base_path=/home/ubuntu/FPGA-Accelerator-for-Recommender-Systems/tf_wide_deep_47_table/python/SavedModel/wide_deep/
```

Open another terminal, send inference request to the TF server:

```
conda activate wide_deep
cd FPGA-Accelerator-for-Recommender-Systems/tf_wide_deep_47_table/python/tensorflow_serving/
python client.py --num_tests=256 --server=localhost:9000 --model=wide_deep
```

You can use TensorBoard to observe the profiling information during serving.

