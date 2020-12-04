import tensorflow as tf
import argparse

from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

parser = argparse.ArgumentParser(description='Wide and Deep Model Prediction')

parser.add_argument(
    '--checkpoint_dir', type=str, default="../model/wide_deep/model.ckpt-237.index",
    help='Model checkpoint dir for evaluating.')

FLAGS, unparsed = parser.parse_known_args()


print_tensors_in_checkpoint_file(file_name=FLAGS.checkpoint_dir, tensor_name='', all_tensors=False)

# Create a new model instance
model = create_model()
# Restore the weights
model.load_weights('./checkpoints/my_checkpoint')