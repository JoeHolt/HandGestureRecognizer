## CS 639 Semester Project - Hand Gesture Recognition

Over the course of the past semester, I have been working to build a Hand Gesture Recognition application. The application uses computer vision techniques we learned about in the course to segment a hand from a webcam stream and make a gesture prediction based on the hand. In the current build, the application can perform system actions when it recognizes a gesture. This web page discusses the motivation, approach, implementation, result and problems encountered while working on the project. Feel free to email me with any questions or concerns.

### Motivation

The primary reason I choose this topic is because I am interested in how artificial intelligence and computer vision can be used to make computers more accessible to those with disabilities. Many people with disabillities are not able to interact with a computer in the traditional way we expect (ie mouse and keyboard). While the current version of the project only works with common hand gestures, my goal long term was to make it expandable such that one day those with disabilities will be able to use it.

On top of this, I also picked this topic because I thought it would be a great opportunity to work with a variety of machine learning and computer vision topics we learned about over the course of the semester, including but not limited to:

- Image Segmantion
- Object recognition
- Neural Networks
- CV workflows

### Approach

My approach involved a variety of steps:

1. Launch the primary application. This starts recording an image stream from the webcam
2. Grab the most recent frame and save it to memory
3. Segment the user's hand from the background, creating a mask of the hand
4. Run the user's hand through a model trained to classify the gesture on the hand
5. Interpret the model results and preform an action based on the results

The implementation section of this site goes into more detail on how each of these steps works. At a high level, the image segmentation process uses a binary mask against a static background grabbed from the initial frame. The model is a pre-trained ResNet model trainsfer trained on a dataset built from the Kaggle Hand Gesture Recognition dataset. 

### Implementation

#### Primary Application
The primary applicaiton for my program is a realtively simple desktop application written in Python. On launch, the application starts a webcam stream where it constantly takes inputs from the camera and runs the frames through the computer vision pipeline described bellow. The application also handles the modification of system behavior based on gestures, and provides a glimpse into the behind-the-secnes by showing the segmented image and and the current prediction + accuracy. 

#### Image Segmentation
The image segmentation process uses a relatively simple binary mask system and works as follows:

1. When the application launches, the image segmenter assumes there is no person or limbs in view
2. The segmenter saves the first frame that comes through the webcam stream for later use
3. On each frame following the first frame, the segmenter creates a binary mask by 'subtracting' the initial frame from the most recent frame
4. To do this, the image is translated to grayscale and logically subtracted from the background
5. Apply the mask to the origional picture, masking out the background and replacing it with black
6. This masked image is then resized and ran through the model
7. A preview of this segmented image can be seen in the bottom right corner of the application

This method works but comes with a variety of drawbacks. Most notably, it is prone to noise and if the camera moves the segmentation is ruined until the application is restarted. Over the spring semester I hope to improve the segmentation algorithm and make it more robust. Long term, I hope to get this running on a smartphone camera.

#### Model / Image Prediction 

##### Model V1 - Simple CNN
The initial model I took was a basic. The model was trained on the raw Kaggle Hand Gesture Recognition dataset, with each image in the set being ran through our image segmentor (with a black static image) and resized to a usable size. The model was relatively simple, made up of 2 convolution layers and 3 fully connected layers. The model provided the following results:

![v1-results](https://github.com/JoeHolt/HandGestureRecognizer/raw/main/mid-semester/curr_results.png)

These results were not great, with an average accuracy around 42%. The movement based classes also had 0% accuracy. When deploying this model to my application, the results were even worse, the model seemed to guess that every single gesture available was an l or thumb. After looking at these results, I knew I need to improve my approach.

##### Model V2 - Deep Learning + Better Data

After the failure of my first model, I looked more at the current state of the art approaches to see if I could learn something from them (more info on this at the bottom of this page). I found that deep learning based models tended to preform significantly better than any of the other approaches. So, with this in mind, I decided I would attempt to transfer train a ResNet-18 model on a similar dataset to that in the first version of the model. I downloaded the pre-trained resnet-18 from torchvision. Once I had the model, I changed the final layer to a simple fully connected linear layer that condensed the outputs to 8 classes. I picked ResNet because I used it over the summer at an internship and had good results with it. 

The primary change I made within the dataset was to completely removing the 'moving' gestures as this was not something I cared to classify (I only care about static gestures). Another change I implemented was, when loading the data, to transform each train image item via random rotation. The idea behind this was that my recognizer could be having trouble due to all hand gestures in the training set being in a similar/set orientation. 

Once I picked my new model and preped my dataset, I trained the model. Unfortunately, I was heavily bounded by compute power so this took ~21 hours for 25 epochs of CPU training with my laptop. After training the model, I saw much better accuracies than before (accuracy over 1004 test images, similar to before, but with 2 less classes):

![v2-results](https://github.com/JoeHolt/HandGestureRecognizer/raw/model-testing/docs/imgs/resnet_acc.png)

As you can see, for nearly every class we had accuracy over 70% and an overall accuracy over 80%. I believe I could have squeezed a little more accuracy out of this model by training for 20-30 more epochs, but I did not have the computer power necessary. I plan on improving this at some point in the spring once i have access to a GPU.

The model was trained using Stochastic Gradient Descent with momentum, Cross Entropy Loss and a learning rate schedueler. 

#### Computer Actions 
The computer actions portion of the project was relatively simple. For each frame ran through the model, the applation either performed and action or didnt. The application preformed and action if the following conditions were met:

1. The given gesture was present in 4 of the last 5 predictions (predictions occur every 6 frames)
2. The given gesture had an accuracy above 70% over the predictions

For my presentation, I implemented two gesture actions, but more could easily be added for each of the other 8 gesture the system can recognize:

1. [Gesture 1] [Action 1]
2. [Gesture 2] [Action 2]

The actions were carried out with the help of the AppleScript and the associated python library. 

### Results
For a live look at the resulting application, take a look at my video presentation bellow:

{% include wiscPlayer.html %}

[Video Link (hosted on cs site)](http://pages.cs.wisc.edu/~holt/demo1080.mp4)

### Future Work

Going forward there a variety of improvements I would like to make. 

Model:
- I would like to try to train the current model for 200 total epochs with a decreasing learning rate to see how much accuracy we can get
- I want to try training a deeper ResNet model (ie 50 or 150) and see those results

Dataset:
- I want to try to find a dataset of hand gestures that are common for people of certain disabilities and train a model based on that. 
- Once I finish doing this, I would like to test this out with people with said disabilities and get their feedback to see if it could help them.

Computer Actions:
- I want to survery people with certain disabilities and ask which computer actions are most important for them, so that this is better suited to those that need it
- I would like to implement a "ghost" version of the application that would run in the background and always be watching (perhaps only for desktop computers initially, due to power limits)

Image Segmentation:
- I think that this area has the most potential work to be done
- I would like to try to improve the current method as much as possible
- I would also like to try to train an object recognition model that would identify where in the frame a hand was, and auto mask out the background
- Ideally, the image segmentation would still work even with a shakey camera

### Current State of the Art

Due to the vast potential applications for a robust noise-agonistic hand gesture recognition system, there has been lots of research on the topic. Some researches have tackled non Computer-Vision approaches with varying degrees of success, but I'll focus on the computer vision applications here.

In a recent [https://www.mdpi.com/2313-433X/6/8/73/pdf](review, Munir Oudah, Ali Al-Naji, and Javaan Chahl) analyzed the current state of modern hand gesture recognition systems. The analysis went over both CV and non-CV methods, but had a focus on CV. 

The paper split the results of various researchers in categories based on the approach. I will highlight the methods that are similar to mind and the given accuracy bellow:

#### Appearance Based Detection
Appearance based detection of hands and gestures using features detected in 2D images. This appraoch is very similar to the one I took in doing this assignment. The current best approach in this category was in a paper "Fang, Y.; Wang, K.; Cheng, J.; Lu, H. A real-time hand gesture recognition method." and was able to recognize 6 hand gestures at 93% accuracy using a Gaussian Model for skin color and segmentation, and a palm-finger configuration algorithm for classificaiton.

#### Motion Based Detection
Motion based detection uses a series of images taken within close proximity of one another to detect the hand and its features, which are then ran through a model for prediction. This method proved to provide beter results than apperance based recognition. In the paper "Pun, C.-M.; Zhu, H.-M.; Feng, W. Real-time hand gesture recognition using motion tracking.Int. J. Comput.Intell. Syst.2011,4, 277–286.", researchers were able to achieve an accuracy of 97.33% for 10 gestures using color and motion tracking segmentation with a histogram distribution model.

#### Skeleton Based Recognition
Skeleton based recognition involves mapping a skeleton of the hand on top of a hand in the training dataset and from arbitrary images. The skeltal structure is then used as the features that are ran through a model for hand gesture recognition. In the paper "Devineau, G.; Moutarde, F.; Xi, W.; Yang, J. Deep learning for hand gesture recognition on skeletal data.In Proceedings of the 2018 13th IEEE International Conference on Automatic Face & Gesture Recognition(FG 2018)" researchers were able to classify 14 gestures with 91.28% accuracy using a CNN.

#### Deep Learning Based Recognition 
The area where researchers have seen the most success in terms of hand gesture recognition is with Deep Learning based approaches. These methods provide results with accuracy many percentage points higher than the aforementioned methods. The current champion in this category is the paper "Alnaim, N.; Abbod, M.; Albar, A. Hand Gesture Recognition Using Convolutional Neural Network forPeople Who Have Experienced A Stroke." In this paper, the researchers were able to achieve an accuracy of 100% on the training set and 99% on the test set for 7 different hand gestures. They handled feature extraction using CNN techniques, and the model as a whole was an Adapted Deep Convolutional Neural Network. With this accuracy, the model was likely as good as if not better than a human doing the same task.

#### Depth Based Recognition
Researchers were able to get great results using 3D cameras for hand gesture recognition. I am not going to go into too much detail because it is out of the scope of this project (I dont have a 3d camera), but XBOX Kinect was able to achieve 96% accuracy using this method. 

### Problems

#### Dataset
One of the first problems I had was acquiring a good dataset that could work with my application. I had thought that the initial dataset I picked (the XBOX kinect gesture dataset) would work well, but I eventually found out that the data had 3D/infraed data. As I looked for a different dataset, I found that infared and 3D data was common amoung datasets such as these. 
After refining my search, I eventually found the Kaggle dataset and its relatively simple 2D, 3 channel jpg images. In hindsight, I likely could have done some data munging with the other datasets to get normal images, but this also likely would have taken a long time.

Furthermore, once I found a good dataset, I found it still was not very helpful because the models were biased towards gestures in certain orientations. To fix this problem, I added a random rotation to each item in the training set so that it would not be so biased. I saw this worked pretty well. 

#### Poor Model Accuracy
The first model I created had very poor accuracy across the training set. To solve this problem I did some more research into state-of-the-art solutions similar to mind from the past, and found that many of them used much more complex models that the simple one I had defined. This is what caused me to go with the deeper ResNet model. 


#### Noisey Image Segentation
The image segmentation process I have been using worked decently well but would always have noise that the training data did not (ie messed up croppings). To fix this issue I ended up using a guassian filter like we learned about in class on the mask before applying it to the image. This fixed the issue for the most part. 

#### Poor Performance
I was initially trying to process a prediction for every frame that was provided to me by the webcam stream (which I believe was 24 FPS). When running the application under this senario, the application was laggy and stuttered. After debugging I found that this was due to it taking longer than 1/24th of a second to make a prediction. To fix this issue, I changed the application so that it would only process a prediction every 6 frames, giving the model ~0.25s to process. This improved the performance dramatically. 

