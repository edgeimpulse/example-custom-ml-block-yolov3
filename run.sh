#!/bin/bash
set -e

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

cd $SCRIPTPATH

POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    --epochs) # e.g. 50
      EPOCHS="$2"
      shift # past argument
      shift # past value
      ;;
    --learning-rate) # e.g. 0.01
      LEARNING_RATE="$2"
      shift # past argument
      shift # past value
      ;;
    --data-directory) # e.g. 0.2
      DATA_DIRECTORY="$2"
      shift # past argument
      shift # past value
      ;;
    --out-directory) # e.g. (96,96,3)
      OUT_DIRECTORY="$2"
      shift # past argument
      shift # past value
      ;;
    *)
      POSITIONAL_ARGS+=("$1") # save positional arg
      shift # past argument
      ;;
  esac
done

if [ -z "$EPOCHS" ]; then
    echo "Missing --epochs"
    exit 1
fi
if [ -z "$LEARNING_RATE" ]; then
    echo "Missing --learning-rate"
    exit 1
fi
if [ -z "$DATA_DIRECTORY" ]; then
    echo "Missing --data-directory"
    exit 1
fi
if [ -z "$OUT_DIRECTORY" ]; then
    echo "Missing --out-directory"
    exit 1
fi

OUT_DIRECTORY=$(realpath $OUT_DIRECTORY)
DATA_DIRECTORY=$(realpath $DATA_DIRECTORY)

IMAGE_SIZE=$(python3 get_image_size.py --data-directory "$DATA_DIRECTORY")

# convert Edge Impulse dataset (in Numpy format, with JSON for labels into something YOLOv3 understands)
python3 -u extract_dataset.py --data-directory $DATA_DIRECTORY --out-directory /data/converted

# Disable W&B prompts
export WANDB_MODE=disabled

# Disable ONNX export during training
sed -i -e "s/ONNX_EXPORT = True/ONNX_EXPORT = False/" /app/yolov3/models.py

cd /app/yolov3
# train:
#     --freeze 10 - freeze the bottom layers of the network
#     --workers 0 - as this otherwise requires a larger /dev/shm than we have on Edge Impulse prod,
#                   there's probably a workaround for this, but we need to check with infra.
python3 -u train.py --img $IMAGE_SIZE \
    --epochs $EPOCHS \
    --data /data/converted/data.data \
    --cfg /data/converted/yolov3-tiny.cfg \
    --weights /app/yolov3-tiny.pt \
    --name yolov3_results \
    --cache
echo "Training complete"
echo ""

mkdir -p $OUT_DIRECTORY

# Copy pytorch model
cp weights/last_yolov3_results.pt $OUT_DIRECTORY/model.pt

echo "Converting to darknet weights..."
python3  -c "from models import *; convert('/data/converted/yolov3-tiny.cfg', '/app/yolov3/weights/last_yolov3_results.pt')"
cp weights/last_yolov3_results.weights $OUT_DIRECTORY/model.weights
echo "Converting to darknet weights OK"
echo ""

echo "Converting to ONNX..."
cd /scripts/darknet/yolo
python3 convert_to_pytorch.py custom
python3 convert_to_onnx.py custom --image-size $IMAGE_SIZE
echo "Converting to ONNX OK"
echo ""

# the onnx file is now in /app/yolov3/weights/last_yolov3_results.onnx

# export as f32
echo "Converting to TensorFlow Lite model (fp16)..."
python3 /scripts/conversion.py --onnx-file /app/yolov3/weights/last_yolov3_results.onnx --out-file $OUT_DIRECTORY/model.tflite
echo "Converting to TensorFlow Lite model (fp16) OK"
echo ""
