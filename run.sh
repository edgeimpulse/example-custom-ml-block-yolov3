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
python3 -u extract_dataset.py --data-directory $DATA_DIRECTORY --out-directory /tmp/data

# Disable W&B prompts
export WANDB_MODE=disabled

cd /app/yolov3
# train:
#     --freeze 10 - freeze the bottom layers of the network
#     --workers 0 - as this otherwise requires a larger /dev/shm than we have on Edge Impulse prod,
#                   there's probably a workaround for this, but we need to check with infra.
python3 -u train.py --img $IMAGE_SIZE \
    --freeze 10 \
    --epochs $EPOCHS \
    --data /tmp/data/data.yaml \
    --weights /app/yolov3-tiny.pt \
    --name yolov3_results \
    --cache \
    --workers 0
echo "Training complete"
echo ""

mkdir -p $OUT_DIRECTORY

# export as f32
echo "Converting to TensorFlow Lite model (fp16)..."
python3 -u export.py --weights ./runs/train/yolov3_results/weights/last.pt --img $IMAGE_SIZE --include saved_model tflite
cp runs/train/yolov3_results/weights/last-fp16.tflite $OUT_DIRECTORY/model.tflite
# ZIP up and copy the saved model too
cd runs/train/yolov3_results/weights/last_saved_model
zip -r -X ./saved_model.zip . > /dev/null
cp ./saved_model.zip $OUT_DIRECTORY/saved_model.zip
cd /app/yolov3
echo "Converting to TensorFlow Lite model (fp16) OK"
echo ""

# export as i8 (skipping for now as it outputs a uint8 input, not an int8 - which the Studio won't handle)
# echo "Converting to TensorFlow Lite model (int8)..."
# python3 -u export.py --weights ./runs/train/yolov3_results/weights/last.pt --data /tmp/data/data.yaml --img $IMAGE_SIZE --include tflite --int8
# cp runs/train/yolov3_results/weights/last-int8.tflite $OUT_DIRECTORY/model_quantized_int8_io.tflite
# echo "Converting to TensorFlow Lite model (int8) OK"
# echo ""
