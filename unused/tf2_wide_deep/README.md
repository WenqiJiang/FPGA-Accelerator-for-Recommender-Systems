# Wide and Deep Learning for Recommender Systems in Tensorflow

Original GitHub Repository: https://github.com/Lapis-Hong/wide_deep

Link to the paper: https://research.googleblog.com/2016/06/wide-deep-learning-better-together-with.html

## Environment
python2

tensorflow >= 2.0

Setup environment with Anaconda:  
```
conda create -n tf2 python=2.7
conda activate tf2  
conda install -c anaconda tensorflow===2.0
conda install -y pyyaml  
```

## Running the code

### Training
You can run the code locally as follows:

```
cd python
python train.py
```
or use shell scripts as follows:
```
cd scripts
bash train.sh
```

### Testing
```
python eval.py
```
or use shell scripts as follows:
```
bash test.sh
```

### TensorBoard

Run TensorBoard to inspect the details about the graph and training progression.

```
tensorboard --logdir=./model/wide_deep
```

## Code Analysis

	• conf -> where all config yaml files are stored
		○ important
			§ model.yaml -> network structure
			§ schema.yaml -> sepcify the schema of the dataset, the first column of each row is the lable (0 / 1, whether the item is clicked), the rest are the features (60 features in total, including continuous and category features)
			§ feature.yaml -> how to process these features
				□ continuous feature
					® can have preprocessing, e.g. normalization
					® reference: tf.feature_column.numeric_column
				□ category feature -> 3 classes
					® hash bucket -> used for stuffs with huge amount of classes, e.g., strings, meaningless ids, etc. These features have too much possibility to be counted, thus use hash function, the hash table size will turn out to be the number of entries of the embedding table
						◊ e.g. adplan_id: 71597; ip_original; 223.73.129.109; galaxy_id: 05858d56b24959964f5e84eb6bf76c71
					® ids -> countable, count total number of id, which will turn out to be the number of entries of the embedding table
						◊ e.g. industry_level1_id: 4
					® vocabularies -> unlike string, with just limited options, count all vocabularies,  which is the number of entries of the embedding table
						◊ e.g. day: 01-31; device_type: [0,1,2]; client_type: ['a','w']
			§ cross_feature.yaml 
				□ some times a feature appears with different features can mean very different stuffs
				□ thus, correlate features by cartesian product; store them into a hash table; build embedding (hash table size = embedding entry number)
				□ e.g., category & location & site:hash_bucket_size:100 
				□ is_deep: whether to include this cross_feature in deep model
				□ notice that the hash bucket size specified in the yaml file will by multiplied by 1000, see lib/read_conf.py
		○ unimportant
			§ data_process.yaml -> setting for spark
			§ train.yaml 
			§ serving.taml
	• data -> extremely unbalanced data! rare positive samples
		○ train -> training set, including lables and features
			§ 6 / 10,000 positive samples
		○ eval -> evaluation set,  including lables and features
			§ 0 / 5,000 positive samples
		○ pred -> evaluation set, no lables, used for model evaluation
			§ 0 / 5,000 positive samples
		○ test -> test set,  including lables and features
			§ 0 / 5,010 samples positive
	• python
		○ train.py -> train & evaluate model
			§ model=build_custom_estimator(model_dir,FLAGS.model_type)
		○ pred.py -> given input features, run prediction
		○ eval.py -> evaluate the model on the test set
		○ lib
			§ build_estimator.py
				□ build_custom_estimator()
					® load data and preprocessing them according to feature.yaml, the processed feature will served as inputs to wide / deep models separately (build_model_columns)
					® return WideAndDeepClassifier (defined in joint.py), which takes wide / deep columns as inputs
				□ _build_model_columns() -> which features to feed into wide / deep models separately
					® some of the features in the schema are not used, only those appeared in feature.yaml are used
					® wide_columns
						◊ all category features -> one hot encoded
							} hash bucket 
								– wide_dim += #bucket size
							} vocabulary
								– wide_dim += #total vocab
							} id
								– wide_dim += #total id
						◊ continuous features -> discretize -> then one hot encoded
							} cut n slices -> n + 1 ranges in total -> encoded as n + 1 dimensional one-hot vectors 
							} wide_dim += n + 1
							} if boundries not specified, then don't use this continuous feature at all
						◊ cross-feature
							} wide_dim += hash_bucket_size
					® deep_columns
						◊ continuous features
							} deep_dim += 1
						◊ category feature
							} hash bucket
								– too many buckets, so these features are encoded into embeddings
								– deep_dim += embedding_size
							} vocabulary
								– since #classes is not much, e.g., 10, still use one-hot inputs, no embeddings 
								– deep_dim += #total vocab
							} id
								– one-hot inputs
								– deep_dim += #total id
						◊ cross-feature
							} similar to hash bucket, use embeddings instead of one-hot
							} deep_dim += embedding_size
				□ joint.py -> wide_deep model
					® WideAndDeepClassifier
						◊ a class which calls _wide_deep_combined_model_fn to construct the neural network
						◊ In this sample dataset, n_classes=2, which uses sigmoid function to judge 0 / 1; in real models otherwise, n_classes > 2 and softmax function is employeed to rank all items
					® _wide_deep_combined_model_fn()
						◊ construct the network
						◊ define dnn_logits and linear_logits first, then combine them (simply add up the results of two models, no logarithm found here)
							} logits=tf.add_n(logits_combine)
				□ dnn.py -> deep model
				□ linear.py -> wide model
	• model -> where the trained model is stored
	• scripts -> bash files for train, test, etc.
		
## Some Concrete Numbers
	• Input dimension
		○ wide_dim = 12,714,809 (10,556,709 for features; 2,158,100 for cross_features) 
			§ feature_num: 39; cross_feature_num: 31; total_feature_num: 70
			§ thus 70 times of random access in total
		○ deep_dim = 734 (502 for features; 232 for cross_features)
		○ # build_estimator.py breakpoint line 138, 160
		○ CPU for wide model while FPGA for deep model
			§ 48MB space needed for wide model is float32 is employed
			§ however, wide input features are one-hot encoded, thus not all of the weights are needed to be cached, CPU cache may be able to store most of the frequently accessed weights
		○ since CPU + FPGA architecture is needed, preprocessing task (from schema to deep / wide inputs) is naturally assigned to CPU
	• Weights
		○ pred.py line 70
		○ dnn/dnn_1/hiddenlayer_0/...
		○ dnn/dnn_1/logits/... -> output layer
		○ dnn/input_from_feature_columns
			§ e.g. 'dnn/input_from_feature_columns/input_layer/age_bucketized_X_site_X_ugender_embedding/embedding_weights'
			§ embedding weights only for hash_buckets / cross_features, since vocab and id will serve as one-hot vector and will be concatenated with embeddings
		○ linear/linear_model
	• TODO: input / intermediate
	• Batch Norm in the hidden layers
		○ e.g. name_ls=model.get_variable_names()
		025 = {str} 'dnn/dnn_1/hiddenlayer_2/batch_normalization/moving_mean'
		026 = {str} 'dnn/dnn_1/hiddenlayer_2/batch_normalization/moving_variance'

