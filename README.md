# **Traffic Sign Recognition** 

---

Author: Cristian Alonso

Date: 2017-04-16

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

[image1]: ./examples/barchart_before.png "Percentage of signals before"
[image2]: ./examples/normalization.png "Normalization"
[image3]: ./examples/random_transform.png "Random modifications"
[image4]: ./examples/barchart_after.png "Percentage of signals after"
[image5]: ./examples/lenet.png "Lenet-5 Architecture"
[image6]: ./examples/signal3.png "Traffic Sign 1"
[image7]: ./examples/signal1.png "Traffic Sign 2"
[image8]: ./examples/signal4.png "Traffic Sign 3"
[image9]: ./examples/signal5.png "Traffic Sign 4"
[image10]: ./examples/signal2.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### README

Here is a link to my [project code](https://github.com/cralonsov/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

Here is an exploratory visualization of the data set. I plotted the data into a bar chart to look how many times each signal image is repeated. As it can be seen here, there are really big differences between some signals and anothers.

![alt text][image1]

For example, the **Speed limit (20km/h)** signal (first_index) has only 180 images for training but the **Speed limit (50km/h)** (third_index) has 2010! With these differences, problems may appear later when the car is not able to identify a signal because the neural network was not trained with enough data.


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to normalize the dataset so that the average brigthness of the images is removed.

When looking at the images, it can be seen that the dataset has many different illumination conditions. So if we remove the average pixel value per data point, we can focuse more in the concrete details of the pictures.

Firstly, I was trying to add random noise to the dataset and then normalizing it, but it seemed to be a bad choice as sometimes, rotating the image included a big black region that made the CNN to increase the focus in the corners rather than in the signal.

Here is an example of a traffic sign image before and after normalization.

![alt text][image2]

Then I increased the number of the images that were less repeated due to the big differences of examples between each class. 

To do that, I looked which data was under the mean of each class repetitions and increased that images copying each one by that ratio.

After that step, I augmented the whole dataset by multiplying it by three and then adding to each image random noise, which included increasing it size (but keeping 32x32 pixels), rotating them between -15 and 15 degrees, adding random sheering and random perspective.

Here is an example of an original image and an augmented image:

![alt text][image3]

Suming up, the difference between the original data set and the augmented data set is the following:

* The differences between classes has been reduced, as it can be seen in the barchart
* The images have been normalized to remove the differences in brightness
* The size of the model has been multiplied by 3
* The images have been geometricaly transformed to include more examples in the dataset

![alt text][image4]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

![alt text][image5]

The model used for this project is Lenet-5, a convolutional network by Yann Lecun that was originally designed for handwritten and machine-printed character recognition digitized in 32x32 pixel images.

The reason for choosing this model is that as we have our dataset preprocessed to have a size of 32x32 pixels, the model may be a good starting point.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6	 				|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16	 				|
| Flatten				| 400 outputs									|
| Fully connected		| 120 outputs 									|
| RELU					|												|
| Fully connected		| 84 outputs 									|
| RELU					|												|
| Fully connected		| 10 outputs 									|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the following parameters:

##### Number of epochs
An epoch is a forward an dackward pass of the whole dataset used to increase the accuracy of the model without requiring more data.

For my model, I used **100** as a number of epochs to give the model enough time to reach the final accuracy.

##### Batch size
I used a batch size of **128**. I used it because it is recommended to use an small batch size to improve results in speed, memory use and accuracy. 

A common size of the batches are 32, 64, 128, 256, 512...

##### Learning rate
As we are using a big dataset, summing up all the weights in each step can lead to really large updates that would make the gradint descent diverge. To compensate this, we need to use a small learning rate, which I saw that the best one was **0.001**.

Learning rates are tipically in the range of 0.01 to 0.0001. 

##### Optimizer
The model used for the optimizer is the [Adaptative Moment Estimation (Adam)](https://arxiv.org/abs/1412.6980), which keeps separate learning rates for each weight as well as an exponentially decaying average of previous gradients.

It is reputed to work well for both sparse matrices and noisy data.


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* Training set accuracy of **0.990**
* Validation set accuracy of **0.934** 
* Test set accuracy of **0.925**

The architecture chose was, as stated before, Lenet-5. It was firstly designed to classify hand-written data but it seems to work really well with traffic signals as they are mostly numbers and arrows.

The most difficult part has been dealing with the images, as there were many that were difficult to classify even for the human eye due to the poor light conditions.

In respect to the model architecture, I tried with different learning rates and number of epochs.

When the first one was increased, the accuracy changed completely from one batch to the next one. And when I reduced it, the accuracy did not barely increase after each batch.

I tried different number of epochs, from 10 to 100 with different results. A low number of epochs will not give enough time to the CNN to reach the final accuracy. However, if we used a big number for this, the CNN would take a huge amount of time to validate the model.

And as it can be seen in the [project code](https://github.com/cralonsov/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb), the accuracy increases with big steps until the 10-20th epoch and the it does it little by little.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image6] ![alt text][image7] ![alt text][image8] 
![alt text][image9] ![alt text][image10]

For the human eye, I think that the most difficult signals to classify would be the two speed limit signals due to the poor light conditions that they have. The 70 km/h signal borders look like part of the landscape. 

I also think that the "Road Work" signal would be difficult to classify because the signal background looks like it has the same color as the sky.

That being said, after normalizing the images, it can be seen that these problems commented above, dissapear because this operation let the CNN focuse more in the details as it removes the brightness of the image.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Priority road   		| Priority road									| 
| Speed limit (70km/h)	| *Speed limit (30km/h)*						|
| Turn left ahead		| Turn left ahead								|
| Speed limit (60km/h)	| Speed limit (60km/h)			 				|
| Road Work				| Road Work										|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. And as it was predicted, the 70 km/h signal was the one that was unproperly classified. 

It can be seen in the barchart of the IPython Notebook as the signal predicted, Speed limit (30km/h), is among the most repeated signals in the training set, so even though this is a bad prediction, it can be understandable.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. 

The code for making predictions on my final model is located in the 137th cell of the Ipython notebook.

For the first image, the model is completely sure that it was a *Priority Road* signal. This one was probably easy because it is probably the only one in the dataset with these colors and shape. The top five soft max probabilities were:

| Probability (Image1) 	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.000000				| 12 - Priority Road							| 
| 0.000000				| ---											|
| 0.000000				| ---											|
| 0.000000				| ---											|
| 0.000000				| ---											|


The second image was the one incorrectly classified. The predicted signal was with 99.9% of accuracy *Speed Limit (30 km/h)*, while the real one was *Speed Limit (70 km/h)*. 

Even though that the CNN "was completely sure' that the signal was 30 km/h, it has the lowest accuraccy of all the predictions (99.82%). Then, the the second prediction, with 0.17 % is the real signal.

Looking at the predictions, it gives us the impression that we have to increase the dataset of the speed limit signals. This is because they are very similar and we have 8 of them with the same shape, color and always sharing the last number (0).

| Probability (Image2) 	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.998289     			| 1 - Speed limit (30km/h)						| 
| 1.71e-03				| 4 - Speed limit (70km/h)						|
| 1.42e-07				| 0 - Speed limit (20km/h)						|
| 4.18e-08    			| 7 - Speed limit (100km/h)						|
| 7.74e-12			    | 8 - Speed limit (120km/h)						|


The rest of the images, as it can be seen in the following tables, were sucessfully classified:

| Probability (Image3) 	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.000000				| 34 - Turn left ahead							| 
| 0.000000				| ---											|
| 0.000000				| ---											|
| 0.000000				| ---											|
| 0.000000				| ---											|


| Probability (Image4) 	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.000000				| 3 - Speed limit (60km/h)						| 
| 2.84e-14				| 5 - Speed limit (80km/h)						|
| 1.99e-16				| 15 - No vehicles								|
| 1.08e-18    			| 2 - Speed limit (50km/h)						|
| 6.96e-20			    | 1 - Speed limit (20km/h)						|


| Probability (Image5) 	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.000000				| 25 - Road work								| 
| 1.83e-25				| 30 - Beware of ice/snow						|
| 3.72e-35				| 22 - Bumpy Road								|
| 4.01e-37    			| 29 - Bicycles crossing						|
| 0.000000				| ---											|