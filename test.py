import numpy as np
from PIL import Image
import torchvision.transforms as T
import torch
from tensorflow import keras
import os


def imp(path):
    original = load_img(path, target_size=(224, 224))

    print('PIL image size', original.size)
    plt.imshow(original)
    plt.show()

    numpy_image = img_to_array(original)

    plt.imshow(np.uint8(numpy_image))
    plt.show()
    print('numpy array size', numpy_image.shape)

    image_batch = np.expand_dims(numpy_image, axis=0)

    print('image batch size', image_batch.shape)
    plt.imshow(np.uint8(image_batch[0]))

    return image_batch


def img(image_batch):
    processed_image = vgg16.preprocess_input(image_batch.copy())
    predictions = vgg_model.predict(processed_image)
    label_vgg = decode_predictions(predictions)
    for prediction_id in range(len(label_vgg[0])):
        print(label_vgg[0][prediction_id])


def run(path):
    for file in os.listdir(path):
        if not file.endswith('.jpg'):
            continue
        im_batch = imp(os.path.join(path, file))
        img(im_batch)


if __name__ == "__main__":
    from tensorflow.keras.applications import (
        vgg16,
        resnet50,
        mobilenet,
        inception_v3
    )

    vgg_model = vgg16.VGG16(weights='imagenet')
    inception_model = inception_v3.InceptionV3(weights='imagenet')
    resnet_model = resnet50.ResNet50(weights='imagenet')
    mobilenet_model = mobilenet.MobileNet(weights='imagenet')

    import matplotlib.pyplot as plt
    from tensorflow.keras.preprocessing.image import load_img
    from tensorflow.keras.preprocessing.image import img_to_array
    from tensorflow.keras.applications.imagenet_utils import decode_predictions

    # assign the image path for the classification experiments
    filepath = 'sample_data/images/'
    run(filepath)