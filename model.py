import csv
import itertools
import numpy as np
from skimage import io

def read_data(filename, skip_header = False):
    with open(filename) as csvfile:
        datareader = csv.reader(csvfile)
        
        if skip_header:
            next(datareader)
            
        for row in datareader:
            yield row

def look_ahead(A, num = 5): 
    """
    This will return a vector of the next 5 for each value
    """
    length = len(A)
    return np.dstack([A[i:length-num+i] for i in range(0, num)])[0]

def get_data(data_dir = "./data/", skip_header = False, split=0.8):
    """
    Load the data, augment the steering angle (see 'look_ahead'), and shuffle
    """
    # params
    log_path = data_dir + "driving_log.csv"
    
    def load_image(name):
        filename = name.split('/')[-1]
        current_path = data_dir + "IMG/" + filename
        return io.imread(current_path)
            
    rows = np.array([(load_image(row[0]), load_image(row[1]), load_image(row[2]), float(row[3])) for row in read_data(log_path, skip_header)])
    S = look_ahead([row[3] for row in rows])
    
    train_len = int(split * len(S))
    perms = np.random.permutation(len(S))
    
    return (rows[perms[0:train_len]], S[perms[0:train_len]]), (rows[perms[train_len:]], S[perms[train_len:]])
    
def generate_data(data, batch_size = 128, correction = 0.21, train = True):
    """
    generator for the data
    """
    t, v = data
    if train:
        rows, S = t
    else:
        rows, S = v
        
    images = []
    measurements = []
    
    for i in range(len(rows)):
        row = rows[i]
        steering_center = S[i]
        # create adjusted steering measurements for the side camera images
        steering_left = steering_center + correction
        steering_right = steering_center - correction

        steerings = [steering_center, steering_left, steering_right]

        for p in range(3):
            image = row[p]
            measurement = steerings[p]
            # original image
            images.append(image)
            measurements.append(measurement)

            if len(images) % batch_size == 0 and len(images) > 0:
                yield np.array(images), np.array(measurements)
                images = []
                measurements = []

            # flipped image
            flipped_image = np.fliplr(image)
            images.append(flipped_image)
            measurements.append(-measurement)

            if len(images) % batch_size == 0 and len(images) > 0:
                yield np.array(images), np.array(measurements)
                images = []
                measurements = []

def combine(*generators):
    for generator in generators:
        for res in generator:
            yield res
                    
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers import Conv2D, Cropping2D
from keras.optimizers import Adam

model = Sequential()

# normalization and cropping
model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(160,320,3))) # normalization
model.add(Cropping2D(cropping=((70,25), (0,0))))

# convolution
model.add(Conv2D(18, (3,3), padding="valid", activation="elu", use_bias=False))
model.add(Conv2D(24, (5,5), strides=(2, 4), padding="valid", activation="elu", use_bias=False))
model.add(Conv2D(48, (5,5), strides=(2, 4), padding="valid", activation="elu", use_bias=False))
model.add(Conv2D(64, (3,3), strides=(2, 2), padding="valid", activation="elu", use_bias=False))
model.add(Conv2D(96, (3,3), strides=(2, 2), padding="valid", activation="elu", use_bias=False))

# final
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(256, activation='elu', use_bias=False))
model.add(Dropout(0.5))
model.add(Dense(128, activation='elu', use_bias=False))
model.add(Dense(5, use_bias=False))

# compile model.
model.compile(loss="mse", optimizer=Adam(lr=0.001))
model.summary()

# train model.

t1 = get_data("./data0616/", False)
t2 = get_data("./data0721/", False)

validation = itertools.cycle(combine(generate_data(t1, train = False), generate_data(t2, train = False)))
train = itertools.cycle(combine(generate_data(t1), generate_data(t2)))

model.fit_generator(train, steps_per_epoch = 220, validation_data = validation, validation_steps = 55, epochs=10)

from keras.utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=False)

# save model.
model.save('model.h5')