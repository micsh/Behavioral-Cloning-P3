import csv
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

def create_data(data_dir = "./data0616/", correction = 0.18, skip_header = False):
    # params
    log_path = data_dir + "driving_log.csv"
    images = []
    measurements = []
    
    rows = [row for row in read_data(log_path, skip_header)]
    S = look_ahead([float(row[3]) for row in rows])

    for i in range(len(S)):
        steering_center = S[i]
        row = rows[i]
        # create adjusted steering measurements for the side camera images
        steering_left = steering_center + correction
        steering_right = steering_center - correction

        steerings = [steering_center, steering_left, steering_right]
        throttle = row[6]

        for p in range(3):
            filename = row[p].split('/')[-1]
            current_path = data_dir + "IMG/" + filename
            image = io.imread(current_path)
            measurement = steerings[p]
            # original image
            images.append(image)
            measurements.append(measurement)
            # flipped image
            flipped_image = np.fliplr(image)
            images.append(flipped_image)
            measurements.append(-measurement)
            
    return images, measurements

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
import tensorflow as tf
model.add(Dense(5, use_bias=False))



# compile model.
model.compile(loss="mse", optimizer=Adam(lr=0.0001))
model.summary()

# train model.

# data from track-one
images1, measurements1 = create_data("./data0616/", 0.21, False)
# data from track-two
images2, measurements2 = create_data("./data0721/", 0.21, False)
    
X_train = np.asarray(images1 + images2)
y_train = np.asarray(measurements1 + measurements2)
print("training data shape:", X_train.shape)

model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    shuffle=True,
    epochs=10)

from keras.utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=False)

# save model.
model.save('model.h5')