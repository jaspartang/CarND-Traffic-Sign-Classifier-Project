# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./train_examples/1x.png "Traffic Sign 1"
[image5]: ./train_examples/2x.png "Traffic Sign 2"
[image6]: ./train_examples/3x.png "Traffic Sign 3"
[image7]: ./train_examples/4x.png "Traffic Sign 4"
[image8]: ./train_examples/5x.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/jaspartang/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 37499
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because that color should not make a difference for the classifier (a human can detect the signs using other features than color).

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because that having a wider distribution in the data would make it more difficult to train using a singlar learning rate. 

I used the techniques augment the dataset, this help to impove the accuracy, increased the accuracy about 2%.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:
 
| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 normalized grayscale image 			| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16    				|
| Flatted		      	| outputs 400					   		|
| Fully connected		| outputs 84   									|
| RELU					|												|
| Dropout				| probability 0.5								|
| Fully connected		| outputs 43 classes   							|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I tried many different approaches with the LeNet architecture. Finally, when switching to the new model LeNet2(Modified LeNet):

BATCH_SIZE = 128 - I read an article from Yann Lecun about not using batch size over 32 because in some situations it may not converge to the true minima. On the other hand, I wanted to speed things up a bit, so 64 was the compromise.

optimizer - AdamOptimizer because it is good enough for this purpose.

EPOCHS = 60 - I saw that there is no significant gain in training more epochs. It kind of reached a plateu after that.

learning rate = 0.0009 - in the new network, this learning rate did the job. In the LeNet architecture I tried reducing the rate in higher accuracy range but it did not suffice to reach 97.4%.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 97.4%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

LeNet, because the problem seemed similar the the MNIST problem. Atleast for getting >95% accuracy.

* What were some problems with the initial architecture?

It seemed not deep enough, perhaps not having enough parameters to cope with the wider set of 43 classes and variety. 

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

 Once switching to the new model, validation set and test set were much closer and I immediatly recieved satisfactory results.

* Which parameters were tuned? How were they adjusted and why?

a constant learning rate of 1e-3 was satisfactory. 

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

convolutional layers are very good with picture problems because they work on a small subset of the picture, learns simple things from it, and adds complexity as each layer is being added. 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The Accuracy is 100%, maybe the pictures used for example is easier to be classified.
In the future, I will download much more pictures, test the model and dive into it deeper.
