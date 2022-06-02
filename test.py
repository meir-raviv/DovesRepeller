import numpy as np
from PIL import Image
import torchvision.transforms as T
import torch

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
    filename = 'images/cat.jpg'
    # load an image in PIL format
    original = load_img(filename, target_size=(224, 224))
    print('PIL image size', original.size)
    plt.imshow(original)
    plt.show()
    # convert the PIL image to a numpy array
    # IN PIL - image is in (width, height, channel)
    # In Numpy - image is in (height, width, channel)
    numpy_image = img_to_array(original)
    plt.imshow(np.uint8(numpy_image))
    plt.show()
    print('numpy array size', numpy_image.shape)
    # Convert the image / images into batch format
    # expand_dims will add an extra dimension to the data at a particular axis
    # We want the input matrix to the network to be of the form (batchsize, height, width, channels)
    # Thus we add the extra dimension to the axis 0.
    image_batch = np.expand_dims(numpy_image, axis=0)
    print('image batch size', image_batch.shape)
    plt.imshow(np.uint8(image_batch[0]))
