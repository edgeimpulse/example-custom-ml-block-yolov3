import numpy as np
import argparse, math, shutil, os, json, time
import onnx
from onnx_tf.backend import prepare # https://github.com/onnx/onnx-tensorflow
import tensortflow as tf

parser = argparse.ArgumentParser(description='YOLOv3 model to TFLite')
parser.add_argument('--data-directory', type=str, required=True)
parser.add_argument('--onnx-file', type=str, required=True)
parser.add_argument('--out-file', type=str, required=True)

args = parser.parse_args()

# Load data to get image shape
X_train = np.load(os.path.join(args.data_directory, 'X_split_train.npy'), mmap_mode='r')
image_width, image_height, image_channels = list(X_train.shape[1:])

# Load the ONNX model
onnx_model = onnx.load(args.onnx_file)

# Check that the IR is well formed
onnx.checker.check_model(onnx_model)

# Now do ONNX => TF
tf_model_path = '/tmp/savedmodel'
tf_rep = prepare(onnx_model)
tf_rep.export_graph(tf_model_path)

# TF => TFLite
converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
tflite_model = converter.convert()

# Save the model
with open(args.out_file, 'wb') as f:
    f.write(tflite_model)
