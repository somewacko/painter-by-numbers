"""
painter/__init__.py

`painter` module for training and testing models.

"""

import csv
import os
import random

from heraspy.model import HeraModel
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Dense, merge
from keras.models import Model
from keras.preprocessing import image as keras_image
import numpy as np
from PIL import Image

import resnet50


IMAGE_SIZE = (224, 224)


def build_model():
    """
    Builds a distance metric model using ResNet50 as a base.

    Returns:
        keras.Model, distance metric model.
    """

    resnet_model = resnet50.ResNet50(weights=None)

    shared_model = Model(
        input  = resnet_model.input,
        output = resnet_model.get_layer('flatten_1').output,
    )

    input_a = Input(shape=shared_model.input_shape[1:], name='input_a')
    input_b = Input(shape=shared_model.input_shape[1:], name='input_b')

    model_a = shared_model(input_a)
    model_b = shared_model(input_b)

    # Try just merging and do a logistic regression for now, nothing fancy
    merged = merge([model_a, model_b], mode='concat', concat_axis=-1)
    output = Dense(1, activation='sigmoid')(merged)

    model = Model(input=[input_a, input_b], output=output)

    model.compile(optimizer='adagrad', loss='binary_crossentropy')

    return model


def parse_csv(data_dir):
    """
    Parses a .csv file with training data info into dicts.

    Args:
        data_dir (str): Directory where the data lives.
    Returns:
        dict, information about the training data.
    """

    entries = list()

    with open(os.path.join(data_dir, 'train_info.csv'), 'r') as csvfile:
        entries = [row for row in csv.DictReader(csvfile)
                if os.path.exists(os.path.join(data_dir, 'train', row['filename']))]

    if not entries:
        raise RuntimeError("No entries in .csv file found!")

    # TODO: Partition some artists as validation data

    entries_by_artist = dict()
    for entry in entries:
        if entry['artist'] not in entries_by_artist:
            entries_by_artist[ entry['artist'] ] = list()
        entries_by_artist[ entry['artist'] ].append(entry)

    entries_by_style = dict()
    for entry in entries:
        if entry['style'] not in entries_by_style:
            entries_by_style[ entry['style'] ] = list()
        entries_by_style[ entry['style'] ].append(entry)

    artists = [x for x in entries_by_artist.keys()]
    styles  = [x for x in entries_by_style.keys()]

    return {
        'entries'   : entries,
        'by_artist' : entries_by_artist,
        'by_style'  : entries_by_style,
        'artists'   : artists,
        'styles'    : styles,
    }


def pairs_for_entry(entry, entries_by_artist, entries_by_style, artists, data_dir):
    """
    Picks positive and negative pairs to use for a given entry.

    Picks two images from the same artist randomly then picks two negative cases,
    one randomly, and another from the same style (a "hard" case).

    Args:
        entry (dict): Entry to get pairs for.
        entries_by_artist (dict): All entries, with artists as keys.
        entries_by_style (dict): All entries, with styles as keys.
        artists (list): All artists in the set.
        data_dir (str): Directory to where the data lives.
    Returns:
        dict, a positive pair.
        dict, another positive pair.
        dict, a "hard" negative case from the same style.
        dict, another negative case, selected randomly.
    """

    artist = entry['artist']

    # TODO: Refactor this because it's kinda gross

    # Pick 2 positive pairs
    positive_pair1 = None
    while not positive_pair1:
        random_entry = random.choice(entries_by_artist[artist])
        if random_entry['filename'] != entry['filename']:
            positive_pair1 = random_entry

    positive_pair2 = None
    while not positive_pair2:
        random_entry = random.choice(entries_by_artist[artist])
        if random_entry['filename'] != entry['filename'] and \
                random_entry['filename'] != positive_pair1['filename']:
            positive_pair2 = random_entry

    # Pick a "hard" negative pair from the same style of painting
    negative_pair_hard = None
    while not negative_pair_hard:
        style = entry['style']
        possible_entries = [x for x in entries_by_style[style]
                            if x['artist'] != entry['artist']]
        if possible_entries:
            negative_pair_hard = random.choice(possible_entries)
        else:
            #print("Warning: {} is its own style for artist {}".format(
            #    style, entry['artist']))
            random_artist = random.choice(artists)
            if random_artist != artist:
                negative_pair_hard = random.choice(entries_by_artist[random_artist])

    # Pick a random negative pair
    negative_pair = None
    while not negative_pair:
        random_artist = random.choice(artists)
        if random_artist != artist:
            random_entry = random.choice(entries_by_artist[random_artist])
            if random_entry['filename'] != negative_pair_hard['filename']:
                negative_pair = random_entry

    return positive_pair1, positive_pair2, negative_pair_hard, negative_pair


def load_image(image_path, image_size, augment=False):
    """
    Load an image and format it so it can be used as an input to the network.
    Augment the image with random crops/flips if needed.

    Args:
        image_path (str): Path to the image to load.
        image_size (tuple<int>): Target size of the image.
    Args (optional):
        augment (bool): Whether or not to augment data.
    Returns:
        numpy.array, the loaded image.
    """

    img = keras_image.load_img(image_path)

    w, h = img.size

    if augment:
        # Crop out random square from painting with a size anywhere from
        # 90-100% of its smallest side
        if h < w:
            s = int( h - h*0.1*np.random.random() )
            tx = int( (w-h)*np.random.rand() )
            ty = int( (h-s)*np.random.rand() )
            img = img.crop((tx, ty, s+tx, s+ty))
        else:
            s = int( w - w*0.1*np.random.random() )
            tx = int( (w-s)*np.random.rand() )
            ty = int( (h-w)*np.random.rand() )
            img = img.crop((tx, ty, s+tx, s+ty))

        # Random horizontal flips
        if np.random.rand() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
    else:
        # Crop out center square
        s = int(np.min(w,h))
        tx = (w-s)/2
        ty = (w-s)/2
        img = img.crop((tx, ty, s+tx, s+ty))

    w, h = img.size
    assert w == h

    img = img.resize(image_size)
    img = keras_image.img_to_array(img)

    return img


def training_data_generator(data_dir, entry_info, batch_size=32, input_size=(3,224,224)):
    """
    Generator to supply the network with training data.

    Args:
        data_dir (str): Directory where the data lives.
        entry_info (dict): Information about entries in the dataset.
    Args (optional):
        batch_size (int): Batch size to use while training.
        input_size (tuple<int>): Input size to the network.
    Yields:
        dict<str:numpy.array>, inputs for the network.
        numpy.array, outputs for the network.
    """

    entries           = entry_info['entries']
    entries_by_artist = entry_info['by_artist']
    entries_by_style  = entry_info['by_style']
    artists           = entry_info['artists']
    styles            = entry_info['artists']

    assert batch_size % 4 == 0

    while True:
        random.shuffle(entries)

        batch_x = {
            'input_a': np.empty((batch_size,)+input_size),
            'input_b': np.empty((batch_size,)+input_size),
        }
        batch_y = np.empty((batch_size,1))

        i = 0

        for entry in entries:

            img_path = os.path.join(data_dir, 'train', entry['filename'])
            if not os.path.exists(img_path) or len(entries_by_artist[entry['artist']]) <= 1:
                raise RuntimeError("{} does not exist or artist {} only has one painting!"
                    .format(img_path, entry['artist']))

            pos1_entry, pos2_entry, negh_entry, negr_entry \
                    = pairs_for_entry(entry, entries_by_artist, entries_by_style, artists, data_dir)

            pos1_path = os.path.join(data_dir, 'train', pos1_entry['filename'])
            pos2_path = os.path.join(data_dir, 'train', pos2_entry['filename'])
            negh_path = os.path.join(data_dir, 'train', negh_entry['filename'])
            negr_path = os.path.join(data_dir, 'train', negr_entry['filename'])

            img1 = load_image(img_path, IMAGE_SIZE, augment=True)
            img2 = load_image(img_path, IMAGE_SIZE, augment=True)
            img3 = load_image(img_path, IMAGE_SIZE, augment=True)
            img4 = load_image(img_path, IMAGE_SIZE, augment=True)

            pos1 = load_image(pos1_path, IMAGE_SIZE, augment=True)
            pos2 = load_image(pos2_path, IMAGE_SIZE, augment=True)
            negh = load_image(negh_path, IMAGE_SIZE, augment=True)
            negr = load_image(negr_path, IMAGE_SIZE, augment=True)

            batch_x['input_a'][i,:,:,:] = img1
            batch_x['input_b'][i,:,:,:] = pos1
            batch_y[i,:] = 1

            batch_x['input_a'][i+1,:,:,:] = img2
            batch_x['input_b'][i+1,:,:,:] = negh
            batch_y[i+1,:] = 0

            batch_x['input_a'][i+2,:,:,:] = img3
            batch_x['input_b'][i+2,:,:,:] = pos2
            batch_y[i+2,:] = 1

            batch_x['input_a'][i+3,:,:,:] = img4
            batch_x['input_b'][i+3,:,:,:] = negr
            batch_y[i+3,:] = 0

            i += 4

            if i >= batch_size:
                i = 0
                yield batch_x, batch_y

        # Skip the last batch. Keras/TensorFlow is complaining that there are
        # less training examples than there are targets, even though it's not
        # true. (It probably is expecting a certain batch size, so sending
        # a new batch size is unexpcted).
        """
        batch_x['input_a'] = batch_x['input_a'][:i,:,:,:]
        batch_x['input_b'] = batch_x['input_b'][:i,:,:,:]
        batch_y = batch_y[:i,:]

        yield batch_x, batch_y
        """


def train(data_dir, model_path, batch_size=32, num_epochs=100, patience=5):
    """
    Trains the model.

    Args:
        data_dir (str): Directory where the data lives.
        model_path (str): Path where the model should be saved.
    Args (optional):
        batch_size (int): 
    """

    model = build_model()

    info = parse_csv(data_dir)

    print("Found {} paintings with {} artists and {} styles".format(
        len(info['entries']), len(info['artists']), len(info['styles'])))

    hera_model = HeraModel({'id': 'painters'}, {'domain': 'localhost', 'port': 4000})

    callbacks = [
        EarlyStopping(monitor='loss', patience=patience),
        ModelCheckpoint(model_path, monitor='loss', verbose=0, save_best_only=True),
        hera_model.callback,
    ]

    # Compute the number of samples for each epoch, shaving off the samples
    # from the last batch
    samples_per_epoch = 4*len(info['entries'])
    samples_per_epoch -= samples_per_epoch % batch_size

    model.fit_generator(
        training_data_generator(
            data_dir,
            info,
            batch_size = batch_size,
            input_size = model.input_shape[0][1:],
        ),
        samples_per_epoch = samples_per_epoch,
        nb_epoch          = num_epochs,
        callbacks         = callbacks,
        verbose           = 1,
    )


def test(data_dir, model_path, output_path):
    """
    """
    pass


