import tensorflow as tf
import numpy as np
import cv2

def process_input(input_details, data):
    """Prepares an input for inference, quantizing if necessary.

    Args:
        input_details: The result of calling interpreter.get_input_details()
        data (numpy array): The raw input data

    Returns:
        A tensor object representing the input, quantized if necessary
    """
    if input_details[0]['dtype'] is np.int8:
        scale = input_details[0]['quantization'][0]
        zero_point = input_details[0]['quantization'][1]
        data = (data / scale) + zero_point
        data = np.around(data)
        data = data.astype(np.int8)
    return tf.convert_to_tensor(data)

def get_features_from_img(interpreter, img):
    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']

    count, width, height, channels = input_shape

    # if channels == width of the image, then we are dealing with channel/width/height
    # instead of height/width/channel
    is_nchw = channels == img.shape[1]
    if (is_nchw):
        count, channels, width, height = input_shape

    if (channels == 3):
        ret = np.array([ x / 255 for x in list(img.flatten()) ]).astype(np.float32).reshape((1, width, height, channels))
    elif (channels == 1):
        rgb_weights = [0.2989, 0.5870, 0.1140]
        img_grayscale = np.dot(img[...,:3], rgb_weights)
        ret = np.array([ x / 255 for x in list(img_grayscale.flatten()) ]).astype(np.float32).reshape((1, width, height, channels))
    else:
        raise ValueError('Unknown depth for image')

    # transpose the image if required
    if (is_nchw):
        ret = np.transpose(ret, (0, 3, 1, 2))

    return ret

def invoke(interpreter, item, specific_input_shape):
    """Invokes the Python TF Lite interpreter with a given input
    """
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    item_as_tensor = process_input(input_details, item)
    if specific_input_shape:
        item_as_tensor = tf.reshape(item_as_tensor, specific_input_shape)
    # Add batch dimension
    item_as_tensor = tf.expand_dims(item_as_tensor, 0)
    interpreter.set_tensor(input_details[0]['index'], item_as_tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return output, output_details

interpreter = tf.lite.Interpreter(model_path="out/model.tflite")
interpreter.allocate_tensors()

img = cv2.imread('data-out/train/images/image00265.jpg')
print('img shape', img.shape)

input_data = get_features_from_img(interpreter, img)
print('input_data', input_data.shape)

output, output_details = invoke(interpreter, input_data, list(input_data.shape[1:]))
print('output_details', output_details)

output0 = interpreter.get_tensor(output_details[0]['index'])
output1 = interpreter.get_tensor(output_details[1]['index'])
print('output0.shape', output0.shape)
print('output1.shape', output1.shape)

blob = output0

# Still WIP analyzing this
step = int(blob.shape[1] / 3)
for oth in range(0, blob.shape[1], step):
    for row in range(blob.shape[2]):       # 13
        for col in range(blob.shape[3]):   # 13
            info_per_anchor = blob[0, oth:oth+step, row, col] #print("prob"+str(prob))
            x, y, width, height, prob = info_per_anchor[:5]
            if (prob < 0): continue

            class_id = np.argmax(info_per_anchor[5:])
            print('row', row, 'col', col, 'x', x, 'y', y, 'w', width, 'h', height, 'prob', prob)


# (1, 21, 5, 5) => 5 is bb attrs + confidence
