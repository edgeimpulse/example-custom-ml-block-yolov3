import numpy as np
import argparse, math, shutil, os, json, time
from PIL import Image

parser = argparse.ArgumentParser(description='Edge Impulse => YOLOv3')
parser.add_argument('--data-directory', type=str, required=True)
parser.add_argument('--out-directory', type=str, required=True)

args = parser.parse_args()

# Load data (images are in X_*.npy, labels are in JSON in Y_*.npy)
X_train = np.load(os.path.join(args.data_directory, 'X_split_train.npy'), mmap_mode='r')
X_test = np.load(os.path.join(args.data_directory, 'X_split_test.npy'), mmap_mode='r')

with open(os.path.join(args.data_directory, 'Y_split_train.npy'), 'r') as f:
    Y_train = json.loads(f.read())
with open(os.path.join(args.data_directory, 'Y_split_test.npy'), 'r') as f:
    Y_test = json.loads(f.read())

image_width, image_height, image_channels = list(X_train.shape[1:])

out_dir = args.out_directory
if os.path.exists(out_dir) and os.path.isdir(out_dir):
    shutil.rmtree(out_dir)

class_count = 0

print('Transforming Edge Impulse data format into something compatible with YOLOv3')

def current_ms():
    return round(time.time() * 1000)

total_images = len(X_train) + len(X_test)
zf = len(str(total_images))
last_printed = current_ms()
converted_images = 0

def convert(X, Y, category):
    global class_count, total_images, zf, last_printed, converted_images

    all_images = []
    images_file = os.path.join(out_dir, category + '.txt')

    for ix in range(0, len(X)):
        raw_img_data = (np.reshape(X[ix], (image_width, image_height, image_channels)) * 255).astype(np.uint8)
        labels = Y[ix]['boundingBoxes']

        images_dir = os.path.join(out_dir, category, 'images')
        labels_dir = os.path.join(out_dir, category, 'labels')
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        img_file = os.path.join(images_dir, 'image' + str(ix).zfill(5) + '.jpg')

        all_images.append(img_file)

        im = Image.fromarray(raw_img_data)
        im.save(img_file)

        labels_text = []
        for l in labels:
            if (l['label'] > class_count):
                class_count = l['label']

            x = l['x']
            y = l['y']
            w = l['w']
            h = l['h']

            # class x_center y_center width height
            x_center = (x + (w / 2)) / image_width
            y_center = (y + (h / 2)) / image_height
            width = w / image_width
            height = h / image_height

            labels_text.append(str(l['label'] - 1) + ' ' + str(x_center) + ' ' + str(y_center) + ' ' + str(width) + ' ' + str(height))

        with open(os.path.join(labels_dir, 'image' + str(ix).zfill(5) + '.txt'), 'w') as f:
            f.write('\n'.join(labels_text))

        converted_images = converted_images + 1
        if (converted_images == 1 or current_ms() - last_printed > 3000):
            print('[' + str(converted_images).rjust(zf) + '/' + str(total_images) + '] Converting images...')
            last_printed = current_ms()

    with open(images_file, 'w') as f:
        f.write('\n'.join(all_images))

convert(X=X_train, Y=Y_train, category='train')
convert(X=X_test, Y=Y_test, category='valid')

print('[' + str(converted_images).rjust(zf) + '/' + str(total_images) + '] Converting images...')

print('Transforming Edge Impulse data format into something compatible with YOLOv3 OK')
print('')

with open(os.path.join(out_dir, 'data.names'), 'w') as f:
    class_names = []
    for c in range(0, class_count):
        class_names.append("class" + str(c))
    f.write('\n'.join(class_names))

with open(os.path.join(out_dir, 'data.data'), 'w') as f:
    f.write('classes=' + str(class_count) + '\n')
    f.write('train=' + str(os.path.join(out_dir, 'train.txt')) + '\n')
    f.write('valid=' + str(os.path.join(out_dir, 'valid.txt')) + '\n')
    f.write('names=' + str(os.path.join(out_dir, 'data.names')) + '\n')
    f.write('backup=backup/\n')
    f.write('eval=coco\n')

cfg = """[net]
# Testing
batch=1
subdivisions=1
# Training
# batch=64
# subdivisions=2
width=""" + str(image_width) + """
height=""" + str(image_height) + """
channels=""" + str(image_channels) + """
momentum=0.9
decay=0.0005
angle=0
saturation = 1.5
exposure = 1.5
hue=.1

learning_rate=0.001
burn_in=1000
max_batches = 500200
policy=steps
steps=400000,450000
scales=.1,.1

[convolutional]
batch_normalize=1
filters=16
size=3
stride=1
pad=1
activation=leaky

[maxpool]
size=2
stride=2

[convolutional]
batch_normalize=1
filters=32
size=3
stride=1
pad=1
activation=leaky

[maxpool]
size=2
stride=2

[convolutional]
batch_normalize=1
filters=64
size=3
stride=1
pad=1
activation=leaky

[maxpool]
size=2
stride=2

[convolutional]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=leaky

[maxpool]
size=2
stride=2

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[maxpool]
size=2
stride=2

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[maxpool]
size=2
stride=1

[convolutional]
batch_normalize=1
filters=1024
size=3
stride=1
pad=1
activation=leaky

###########

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[convolutional]
size=1
stride=1
pad=1
filters=""" + str((4 + 1 + class_count) * 3) + """
activation=linear

[yolo]
mask = 3,4,5
anchors = 10,14,  23,27,  37,58,  81,82,  135,169,  344,319
classes=80
num=6
jitter=.3
ignore_thresh = .7
truth_thresh = 1
random=1

[route]
layers = -4

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[upsample]
stride=2

[route]
layers = -1, 8

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[convolutional]
size=1
stride=1
pad=1
filters=""" + str((4 + 1 + class_count) * 3) + """
activation=linear

[yolo]
mask = 1,2,3
anchors = 10,14,  23,27,  37,58,  81,82,  135,169,  344,319
classes=""" + str(class_count) + """
num=6
jitter=.3
ignore_thresh = .7
truth_thresh = 1
random=1"""

with open(os.path.join(out_dir, 'yolov3-tiny.cfg'), 'w') as f:
    f.write(cfg)
