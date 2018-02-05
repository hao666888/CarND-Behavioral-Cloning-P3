# **Behavioral Cloning** 

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior.
	- The simulator contains two modes, training mode and autonomous mode. We will use training mode to collect data (the turning angles for each image collected by the three cameras on the car). The autonomous mode is used for testing the model we build. 
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road


[//]: # (Image References)

[image1]: ./examples/Nvidia.png "Model Visualization"
[image2]: ./examples/Final_structure.png "Final model"
[image3]: ./examples/center.jpg "center Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

### Model Architecture and Training Strategy

#### 1. Model design

I first implemented a LeNet based model which is able to drive the vehicle for half of the track. Then I implemented a neural network structure designed by [Nvidia](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) as show below: 

![alt text][image1]

The network contains 5 convolution layers and 4 fully connected layer. I used the default layer size. Specically the first layer has 24 channels. The second has 36 channels. The rest have 48, 64 and 64 channels. The kernel size for the first three convolution layers is 5x5. The kernel size for the rest convolution layers is 3x3. The stride for the first three convolution layers 2x2 where that for the rest two convolution layers is 1x1. For the fully connected layers, the first one has a output size of 100. The rest have output size of 50, 10, 1 correspondingly. There is no information about activation layers, pooling layers and dropout layers in the Nvidia paper. I added Relu activation layer after each convolution layer and the first two fully connected layers. This adds nonlinearity to the model. I also added a maxpooling layer with a stride of 2 after the first convolution layer. To further avoid overfitting, a Dropout layer with dropout probability of 0.5 is added after the fifth convolution layer. In addition, a normalization layer is added to before the first convolution layer. During the testing phase, the car is able to drive more smoothly (without frequent moving from one side of the road to the other side of the road). 
(model.py lines 62-88)

#### 2. Attempts to reduce overfitting in the model

The model contains both maxpooling layer dropout layers in order to reduce overfitting 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and also driving in the opposite direction. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use existing successful image classification models. I first used LeNet and found that the car always went off track in the first sharp corner, although the train and validation loss is low. Then I implemented the model suggested by the Nvidia. I improved the model by adding nonlinearity, maxpooling and dropout layers. In addition, I collected more data to tackle the specific turns. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set where 80% data is used for training and 20% is used for validation. 

The model code and data were uploaded to AWS EC2 machine which has GPU. Then I saved and downloaded the trained model to test with the simulator. Then based on how well the car drives on the track, I will update the script or collect more data. 

The process is iterated several times. At the end, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture is here: 

![alt text][image2]


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image3]

Then I turned the car to the opposite direction and recorded two laps. This helps the car from being left turn biased. 

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover if it is run to the side of the road. 

During the testing phase, I noticed that the car run off track in certain corners. Then I recorded the car running those corners in the center lane. 

After the collection process, I had around 40000 sample images. During the training phase, I also used the images took by the left and right camera. For the image tooken by the left camera, the steering angle was adjusted by plus 0.25. For the image tooken by the right camera, the steering angle was adjusted by minus 0.25. 

I finally randomly shuffled the data set. 

I used this training data for training the model. I used 3 epochs since I noticed that validation loss did not decrease beyond 3 epochs. This saved time versus 5 epochs. 

### Results

The vehicle is able to finish the whole lap. A video is included in the repo. Although I tried many different setup and dataset, it is still not perfect. I think it could be better, by cropping the images and performing more preprocessing like changing to grey scale. 
