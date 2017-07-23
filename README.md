# Behavioral-Cloning-P3

### Overview
This is the project submission for Udacity's Self Driving Car Nano Degree: **'*Behavioral Cloning Project*'**.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

---

[//]: # (Image References)

[model]: ./model.png "model"

##### All required files are included, and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md summarizing the work and results

##### To run in autonomous mode, simply run the server as following:

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```
python drive.py model.h5
```

##### model.py

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

#### Model Architecture and Training Strategy

#### Model Architecture

My model consists of a convolution neural network with 5 layers of 3x3 and 5x5 filter sizes and depths between 18 and 96, followed by a fully-connected network.

The model includes ELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. 

![model][model]

##### Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. 

The model was trained and validated on different data sets to ensure that the model was not overfitting. 

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

##### Model parameter tuning

The model used an adam optimizer with a learning rate of 0.0001.

##### Final Model Architecture

Here is a visualization of the architecture:

| Layer (type)              | Output Shape          | Param # |
| -------------             |:-------------         |   -----:|
| lambda_1 (Lambda)         | (None, 160, 320, 3)   | 0 |
| cropping2d_1 (Cropping2D) | (None, 65, 320, 3)    | 0 |
| conv2d_1 (Conv2D)         | (None, 63, 318, 18)   | 504 |
| conv2d_2 (Conv2D)         | (None, 30, 79, 24)    | 10824 |
| conv2d_3 (Conv2D)         | (None, 13, 19, 48)    | 28848 |
| conv2d_4 (Conv2D)         | (None, 6, 9, 64)      | 27712 |
| conv2d_5 (Conv2D)         | (None, 2, 4, 96)      | 55392 |
| flatten_1 (Flatten)       | (None, 768)           | 0 |
| dropout_1 (Dropout)       | (None, 768)           | 0 |
| dense_1 (Dense)           | (None, 256)           | 196864 |
| dropout_2 (Dropout)       | (None, 256)           | 0 |
| dense_2 (Dense)           | (None, 128)           | 32896 |
| dense_3 (Dense)           | (None, 5)              645 |


Total params: 353,685
Trainable params: 353,685
Non-trainable params: 0


#### Solution Design Approach

The overall strategy for deriving a model architecture was to break-down the problem into two conceptual tasks, first, learn the to extract relevant features, i.e. road and lane markings, second, from the features, calculate the desired angle. 

My first step was to use a convolution neural network model similar to the NVIDIA model. I thought this model should be a good starting point, as it worked for them. And also, it was easy to find, as there are many references on the web to that model.

##### Data augmentation

I split my image and steering angle data into a training (80%) and validation set (20%). Each timestamps produces three images, so I'm using all three images, with a correction of 0.21 for right and left. And to further augment the data, each image is also flipped horizontally.

The final step was to run the simulator to see how well the car was driving around track-one. I was pleased to see it drive, even if not perfectly, and stay on the road. After a few trials and tweaks, the car could complete a full track without touching the lane markings. 

Happy with my results so far, I have decided to test the same track with a higher speed (going from 9 mph to 30 mph), it didn't work as well. Now, I could blame it on latency, the drive.py is running on a server in the US, while the simulator is using port-forwarding to connect to it, from across the ocean. But I kept trying to improve it (also it was terrible on track-two).

Frustrating, I have discovered that better test and validation score don't necessarily mean better driving!

After a lot of thought, I have decided to do two things, one, record my self driving track-two and use the extra data for a larger training set. Two, change the model so the output has 5 values instead of 1, and they basically represent the steering angle in the following samples. 

##### Next 5 steering angles

This can be easily done, and will only cost removing the last five samples from the set. How it works is simple, we have the training-log csv file, and with little effort, we can replace each steering angle with a vector of five steering angles, the current one, the one following it, etc... 

##### Why do it?

The model doesn't need to work much harder to learn to produce the 'prediction', the next immediate timestamps shouldn't be harder than the current timestamp. Furthermore, it is likely that the vector of five angles would in most cases produce five identical, or close to identical values (unless the car needs to turn). And also, the loss function calculates the mse, so an mse of the next five timestamps might produce a better estimate and objective. 

#### Final results

The car can drive well on track-one, this is true when trained with the training-set containing track-one data only, and also when trained with both tracks data.

For track-two, trained with just track-one data, the car can drive well, but only until it reaches a sharp curve. When trained with both tracks data, the car can drive well through almost all of track-two. There are a couple of places in track-two where it completely fails.