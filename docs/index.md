## CS 639 Semester Project - Hand Gesture Recognition

Over the course of the past semester, I have been working to build a Hand Gesture Recognition application. The application uses computer vision technique we learned about in the course to segment a webcam stream and make a prediction based on the segmented images. In the current build, the application takes a webcam stream as input and then attempts to guess a hand gesture within the image stream. Based on said image, it will preform various actions on the computer (such as increasing the volume). This web page discusses the motivation, approach, implementation, result and problems encountered while working on the project. Feel free to email me with any questions or concerns.

### Motivation

The primary reason I choose this topic is because I am interested in how artificial intelligence and computer vision can be used to make computers more accessible to those with disabilities that don't allow them to interact with a computer in the traditional way we expect (ie mouse and keyboard). While the current version of the project only works with common hand gestures, my goal was to make it expandable such that one day those with disabilities will be able to use it.

On top of this reason, I also picked this topic because I thought it would be a great opportunity to work with a variety of machine learning and computer vision topics we learned about over the course of the semester, including but not limited to:

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

The implementation section of this site goes into more detail on how each of these steps works. At a high level, the image segmentation process uses a binary mask against a static background grabbed from the initial frame, and the model is a CNN train on data from the Kaggle Hand Gesture Recognition dataset. 

### Implementation

#### Primary Application
The primary applicaiton for my program is a realtively simple desktop application written in Python. On launch, the application starts a webcam stream where it constantly takes inputs from the camera and runs the frames through the computer vision pipeline described bellow. The application also handles the modification of system behavior based on gestures, and provides a glimpse into the behind-the-secnes by showing the segmented image and and the current prediction + accuracy. 

#### Image Segmentation
The image segmentation process uses a relatively simple binary mask system and works as follows:

1. When the application launches, the image segmenter assumes there is no person or limbs in view. It also assumes that the camera will not move until the program closes
2. The segmenter saves the first frame that comes through the webcam stream for later use
3. On each frame following the first frame, the segmenter creates a binary mask by 'subtracting' the initial frame from the most recent frame
4. To do this, the image is translated to grayscale, and then to logical values based on a threshold value
5. This resulting mask is then cropped and resized to a size our model expects

This method works but comes with a variety of drawbacks. Most notably, it is prone to noise and if the camera moves the segmentation is ruined until the application is restarted. Over the summer I hope to improve the segmentation algorithm and make it more robust. Long term, I hope to get this running on a smartphone camera.

#### Model / Image Prediction 

##### Model V1
The initial model I took was a basic CNN. **[TODO: More model info]** The model was trained on the Kaggle Hand Gesture Recognition dataset, with each image in the set being ran through our image segmentor (with a black static image) and resized to a usable size. The model provided the following results:

##### Model V2 - Transfer Learning from ResNet

TODO

#### Computer Actions 
The computer actions portion of the project was relatively simple. For each frame ran through the model, the applation either performed and action or didnt. The application preformed and action if the following conditions were met:

1. The given gesture was present in X of the last Y frames (smoothing out noise from the model)
2. The given gesture had an accuracy above Z% over the last frames 

For my presentation, I implemented two gesture actions, but more could easily be added for each of the other 8 gesture the system can recognize:

1. [Gesture 1] [Action 1]
2. [Gesture 2] [Action 2]

The actions were carried out with the help of the AppleScript and the associated python library. 

### Results
My applicaiton ended up working with the following stats...


Here is my presentation video....


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

