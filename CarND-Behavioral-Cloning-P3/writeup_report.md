
# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed
The framework should be same as most of the CNN, where convolutional layers to reduce/extract the features and fully connected layers to do the deduction. And the input for fc layers should not be too large, so here I reduced the number of features to 4096 (1/40 of the origin).

Since this was a regression task, regularization was important, but as what I had tested, dropout worked not quite well in this task so l2 regularization was chosen.

#### 2. Attempts to reduce overfitting in the model

The model contains l2 regularization in order to reduce overfitting (model.py "Network Architecture" part). 

The model was trained and validated on different data sets, but while training, loss of validation was getting the lowest on the epoch of the third, and later it just vibrated. I thought it made sense because first training data is not perfect, second there should be various good routes for vehicles to run on the road. So it is hard to judge whether model was getting overfitting or not by only loss in validation.

To prove the model was trained well, I can just test model by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py "Training" part).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road on the basic map and driving on the right side on the advanced map. 

I used a combination of center lane driving (both forward & reverse), recovering from the left and right sides of the road (including curves).

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

As I talked in the previous section, training data and validation data had some implicit problems such as not precise and having alternates to get to the finish. So I built a network was similar as "Even More Powerful Network" and just focused on creating good training data.

For details about how I succeeded, see the 3th part "Creation of the Training Set & Training Process" of this section. 

#### 2. Final Model Architecture

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3  							| 
| Lambda     	    | 0-mean-1-std Normalization 	 |
| Cropping     	    | Vertical crop 63 pixels from top & 25 pixels from bottom 	 |
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 74x316x24 	 |
| RELU					|												|
| BatchNormalization					|								|
| Max pooling	      	| 2x2 stride,  outputs 37x158x24 				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 33x154x36 	 |
| RELU					|												|
| BatchNormalization					|								|
| Max pooling	      	| 2x2 stride,  outputs 16x77x36 				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 12x73x48         |
| RELU					|                                                 |
| BatchNormalization					|								|
| Max pooling	      	| 2x2 stride,  outputs 6x36x48                     |
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 4x34x64         |
| BatchNormalization					|								|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 2x32x64         |
| BatchNormalization					|								|
| Fully connected		| (4096, 100)     									|
| RELU           		|                									|
| BatchNormalization					|								|
| Fully connected		| (100, 50)  with l2 regularization: lambda=0.01   									|
| RELU           		|                									|
| BatchNormalization					|								|
| Fully connected		| (50, 10) with l2 regularization: lambda=0.01    									|
| RELU           		|                									|
| BatchNormalization					|								|
| Fully connected		| (10, 1)     									|
| Output		| Steering  									|

#### 3. Creation of the Training Set & Training Process

In the very first time I tried not recording data from the advanced map, and it showed that there would be a risk of overfitting (the vehicle would turn left slightly and went out of the road). So record at least data from both of the tracks should help a lot on generizaiton. 

* Forward on both tracks

I hoped my model having some generization ability, but I can not judge it directly from the validation loss, so I made my data have more variance (bias might be added too) by recording reverse data from both tracks.

* Reverse on both tracks

So far I have collected over 20,000 training data, and the model should be able to run fairly well on basic track (actually it worked fairly well on both), but problems always happened somewhere, like crashing to the left when going through the bridge or not recognizing the sand zone. But it was a good news because at least it could figure out normal situation itself, and the only thing I should do was just to add some data about recovering/ curving/ going through space with special texture.

* Recover on both tracks

Here are all the data I collected for training. For validation, I collected data that was special like not going on the center of the road or somewhere hardly out of the road. For test, I collected another two complete forward going data on both tracks.

* Val: Special/ hard situation
* Test: Another forward on both tracks
