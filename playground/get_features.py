import numpy as np

from keras.preprocessing import image
from keras.models import Model
from keras.applications.vgg16 import VGG16 
from keras.applications.vgg16 import preprocess_input

def generate_features(image_paths):
    """
    Takes in an array of image paths.
    Returns pretrained features for each image.
    :param image_paths: array of image paths.
    :return: array of last-layer activations,
    and mapping from array_index to file_path.
    """

    images = np.zeros(shape=(len(image_paths), 224, 224, 3))

    # Loading a pretrained model.
    pretrained_vgg16 = VGG16(weights='imagenet', include_top=True)

    # Using only the penultimate layer, to leverage learned features.
    model = Model(inputs=pretrained_vgg16.input,
                  outputs=pretrained_vgg16.get_layer('fc2').output)
    

    # We load all our dataset in memory (works for small datasets)
    for i, f in enumerate(image_paths):
        img = image.load(f, target_size=(224, 224))
        x_raw = image.img_to_array(img)
        x_expand = np.expand_dims(x_raw, axis=0)
        images[i, :, :, :] = x_expand

    # Once we've loaded all our images, we pass them to our model.
    inputs = preprocess_input(images)
    images_features = model.predict(inputs)
    return images_features