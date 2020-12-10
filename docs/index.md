## CS 639 Semester Project - Hand Gesture Recognition

Over the course of the semester, I built a Hand Gesture Recognition application. The application uses computer vision techniques to segment a hand from a webcam stream and make a gesture prediction based on said hand. When a gesture is recognized, the application performs system actions (such as changing the volume). This web page discusses the motivation, approach, implementation, results and problems encountered while working on the project. Feel free to email me with any questions or concerns at jpholt2@wisc.edu.

### Motivation

The primary reason I choose this topic is because I am interested in how artificial intelligence and computer vision can be used to make computers more accessible to those with disabilities. Many people with disabilities are not able to interact with a computer in the traditional way we expect (ie mouse and keyboard). While the current version of the project only works with common hand gestures, my goal long term is to make the program expandable such that one day those with disabilities will be able to use it.

On top of this, I also picked this topic because I thought it would be a great opportunity to work with a variety of machine learning and computer vision topics we learned about over the course of the semester, including but not limited to:

- Image Segmantion
- Object recognition
- Neural Networks
- CV workflows

### Approach

The computer vision pipeline for approach looks something like this:

1. Launch the primary application. This starts recording an image stream from the webcam
2. Grab the most recent frame and save it to memory
3. Segment the user's hand from the background, creating a mask of the hand
4. Run the user's masked hand through a model trained to classify the gesture on the hand
5. Interpret the model results and perform an action

The implementation section of this site goes into more detail on how each step of the approach works.

### Implementation

#### Primary Application
The primary applicaiton for my program is a desktop application written in Python. On launch, the application starts a webcam stream where it pulls images from the camera and runs the frames through the computer vision pipeline. The application also handles the modification of system behavior based on gestures, and provides a glimpse into the behind-the-secnes by showing the segmented image and and the current prediction + accuracy on the screen. See the demo linked bellow for a look at the application itself.

#### Image Segmentation
The image segmentation system has two primary phases: calibration and segmentation.

**Calibration:**
During the calibtration phase, the webcam creates a running weighted average of the background frames to later use for separating the forground from the background. I learned about this technique from [this geeksforgeeks article](https://www.geeksforgeeks.org/background-subtraction-in-an-image-using-concept-of-running-average/):

0. Convert each frame to gray scale and apply a Guassian blur
1. For the first 30 frames, do not segment the images
2. Add each frame to a running average using `cv2.accumulateWeighted`. Store this for the segmentation step

The idea is that if we can find the differencee/distance between the weighted background and a new frame, we can easily segment the portions of the new frame that are significantly different from the background based, for some definition of significant. 

**Segmentation**
In my initial approach, I had trouble segmenting my image forground from the background due to noise and slight movements in the camera (see problems section bellow for more info). After doing more research, I found [this article](https://gogul.dev/software/hand-gesture-recognition-p1) by Gogul Ilango who provided a variety of techniques to overcome this problem. The overview of the final process can be seen here:

1. For each frame, take the difference of the frame from the weighted background image
2. Separate the forground and the background using this difference via a threshold mask
3. Find the different masked areas on this difference using the cv2 contours method
4. If a countour is found, return the contour with the largest area. This is (ideally) our hand.

This method works but comes with a variety of drawbacks. Most notably, we must keep the camera still and people/limbs out of vision for 30 frames to calibrate the system. Furthermore, the results get significantly worse if the camera is moved. I would like to improve upon this system in the future (see future work section bellow).

#### Model / Image Prediction 

##### Model V1 - Simple CNN

**Data**
The data used with my first model was a direct copy of the [Kaggle Hand Gesture Recognition dataset](https://www.kaggle.com/gti-upm/leapgestrecog). For each image in this dataset, I prepocessed it as follows:

1. Convert to grayscale
2. Convert to BW via threshold mask
3. Crop to square (only dropping off fully black side columns)
4. Resize to 32x32

**Model**
For my model I created a simple Convolutional Neural Network with 2 convolutional layers and three fully connected layers. I trained the model for around 50 epochs using SGD and a static learning rate. Here are my results:

![v1-results](https://github.com/JoeHolt/HandGestureRecognizer/raw/main/mid-semester/curr_results.png)

These results were not great, with an average accuracy around 42%. The movement based classes also had 0% accuracy. When deploying this model to my application, the results were even worse, the model seemed to guess that every single gesture available was an l or thumb. After looking at these results, I knew I need to improve my approach.

##### Model V2 - Deep Learning + Better Data

**Motivation**
There were three primary reasons I decided to scrap my initial model and start from scratch:

1. My initial model had awful accuracy
2. I knew there were already reliable, tested models out there pre-trained for feature extraction
3. After reasearching state-of-the-art techniques (see section bellow), I found that deep learning techniques were particularly well suited for this specific task

**Data**
I started with the same dataset as before, but made some modifications. The first thing I did was completely remove the data points that were for "moving" gestures. My application is only concerned with static gestures, so these should not have even been included in the first case. The second thing I did was rework how the proprocessing worked:

1. Convert image to gray scale
2. Convert gray image to logical image based on threshold
3. Apply random rotation to each image
4. Resize to 226x226

I choose to keep the BW images because this would be very similar to the output from my image segmentation tool. I choose to add a random rotation because I noticed that when my initial model did work, it only worked in the same orientation as the test data, which was not realistic for my tool. My hope was that with random rotation on the trianing images, the model will be more robust to different inputs.

**Model**
For my model, I choose to transfer-train the pre-trained torchvision ResNet-18 model. I had used a model similar to this over the summer for a project, and thought it would be a good fit here. Ideally I would have used a deeper ResNet model, but I was heavily compute-bound due to the limited power of my laptop (and it's lack of a cuda-GPU). 

After downloading the model, I froze the parameters of all the layers except the last. For the last layer, I choose a simple linear fully connected layer mapping the classifier to the 8 potential output classes. Once my model was setup, I began training. Due to my limited compute power, I was only able to train for 25 epochs (which took 25 hours). The model was trained using Stochastic Gradient Descent with momentum, Cross Entropy Loss and a learning rate schedueler. My results on the test set can be seen bellow:

![v2-results](https://github.com/JoeHolt/HandGestureRecognizer/raw/main/docs/imgs/resnet_acc.png)

As you can see, for nearly every class we had accuracy over 70% and an overall accuracy of around 77%. I believe I could have squeezed a little more accuracy out of this model by training for 20-30 more epochs.

#### Computer Actions 
This portion of the project was simply expanding the primary python application I created to be able to utilize my operating system (I am using macOS) actions. I was able to perform computer actions by calling AppleScript scripts from my python application. An given action (ie volume change) was executed if the following condition was met:

- The given gesture was present in 6 of the last 7 predictions (predictions occur every 6 frames)

In the current implementation of the application (that can be seen in the demo), I mapped two different gestures to the volume controls on my computer:

1. L gesture (thumb and pointer extended): Mute Volume
2. OK gesture (pointer and thumb in O shape, rest of fingers up): Turn volume to 75%


### Results
I am super happy with how my final product turned out. The application meets all the goals I initially set out to meet, and I learned a ton in the process of creating it. The link bellow provides a demo of the application live in action. While watching, ensure your sound is on and make sure to make note of the following:

- The superimposed box is the "aim" box for where the segmentation tool attempts to find the image (note: the actual search area is slightly larger than the box)
- The superimposed box is red until the segmentation tool finishes calibrating
- Bottom left corner showcases the 4 predictions and their respective accuracies at any given time (assuming a valid hand is found)
- The bottom right corner shows the segmented image that is fed to the model (assuming a valid hand is found)
- When an action is performed, the sound bar in the settings pane in the bottom left corner of the screen changes
- When an action is performed, you can hear the music in the background change (audio is from a speaker linked to computer audio)
- When an action is performed, the user is notified of the action via a notification in the top right of the screen

[Application Demo Link (dont forget to turn sound on!)](http://pages.cs.wisc.edu/~holt/demo1080.mp4)

I have also included a link to the presentation I gave the class on 12/10. See the link bellow.

[Presentation Video](http://pages.cs.wisc.edu/~holt/PresentationFinal.mp4)

### Future Work

Going forward there a variety of improvements I would like to make. 

**Model**
There are many modifications to make to the model portion of the project. My most notable curiosities are listed bellow:
0. I would like to make my final layer of the ResNet-18 model a little more complex. Right now it is just a simple linear layer, but if I were to improve upon this I think I could get a higher accuracy. 
1. I would like to train the existing model for around 175 more epochs. I will try this once I have access to a GPU or two
2. I would like to try a deeper resnet model, ie ResNet-152. Again, I will need some compute power first.
3. I would like to try the [EfficientNet-L2-475 + SAM](https://paperswithcode.com/paper/sharpness-aware-minimization-for-efficiently-1) model, the current ImageNet champion
4. Finally, at somepoint I would like to try to fully recreate the [current State-of-the-art](https://www.mdpi.com/2313-433X/6/8/73/pdf) paper where they utilize a Adapted Deep Convolutional Neural Network (ADCNN) model for 99% accuracy.

**Dataset**
My primary concern motivation for finding a better dataset is that my current dataset only supports gestures from able-bodied individuals, which is not my end goal.
1. I would like to find a dataset with RGB image from disabled individuals
2. I would like to survey said individuals for what they would like in an application like mine
3. I would like to try an acquire a depth sensor so I can work with some of the depth-based datasets (and so I can better segment the frames)

**Image Segmentation**
The image segmentaiton tool works, but it is not perfect. I would like to make the following improvements (although I am not sure exactly how at this point in time):
1. Reduce / eliminate calibration time
2. Make the segmentation tool robust to camera movement
3. Reduce noise of system
4. Make system output RBG images (rather than locigcal/grayscale). I think my models could work better under non-logical images.
5. Attempt to use another CNN model to locate and segment the hand, rather than the current method


### Problems
While working on this project, I ran into problems nearly every step of the way. This section discusses some of these problems. 

#### Bad Dataset
One of the first problems I had was acquiring a good dataset that could work with my application. I had thought that the initial dataset I picked (the XBOX kinect gesture dataset) would work well, but I eventually found out that the data had 3D/infraed data. I thought about attempting to generate similar depth data using a normal camera (or my computer + phone camera) and some of the techniques we learned about for depth sensing in class, but this proved to be more trouble than it was worth and I began looking for a different dataset. 

After refining my search, I eventually found the Kaggle dataset linked above. The dataset was made up of relatively simple 2D, 3 channel jpg images. In hindsight, I likely could have done some data munging with the other datasets to get normal images, but this also likely would have taken a long time. 

#### Model Only Worked with Hands in 1 orientation
My initial model had a problem where, when the model worked, it only worked in the hands were in a given orientation (ie hand enters webcam image straight up and down). I noticed the working orientation was the same orientation as the images in the train set. To aliviate this problem, I added a random rotation to each image in the dataset when processing the dataset, as decribed above. 

#### Poor Model Accuracy
The first model I created had very poor accuracy across the training set. To solve this problem I did some more research into state-of-the-art solutions similar to mind from the past, and found that many of them used much more complex models that the simple one I had defined. This is what caused me to go with the deeper ResNet model. This was described above.

#### Noisey Image Segentation
**Background**
My initial image segmentation process looked like this:
1. Record a background frame when the application is initially loaded. This step assumes there is no person in the frame at this time.
2. Convert this image to gray, and then logical via threshold mask
3. For each following frame, transform the frame to grayscale and then logical
4. Subtract the processed background from the processed frame
5. The resulting image is the mask

**Problem 1: This result was super noisey**
I found that often times there was a ton of grainy noise in the masked image. To fix this, I applied a guassian filter as we learned about in class to the image. This worked super well at eliminating the noise and fixed the issue for the most part.

**Problem 2: Not Robust at All to Camera Movement**
This was the largest problem with my approach. If the camera moved at all (be that just one pixel) the segmentaiton was ruined until reseting the tool. I spend alot of time trying to fix this with little success. Eventually I researched some more and found an article describing and showcasing the current segmentation method in the current applicaiton. This method adds the following improvements over my initial idea:

1. Use a weighted accumlated average of the background, rather than just the initial frame
2. Before setting up this average/initial frame, apply the Guaissian blur (I was doing this after subtraction)
3. Limit the segmentater to a small subset of the window, rather than the entire frame
4. Use the findContours method from cv2 to find areas in the resulting mask of varying sizes, the biggest being the hand. (I didnt know this was something that could even be done easily)

All of these changes came together to create a system that was much more robust to movements in the camera and noise. This system is still not perfect (if you move the camera alot, the system still breaks), but it is much better than before. My primary goal as I continue to work on this project next semester is to improve this system.

#### Poor Performance
In my initial build of the application, the application was super slow and laggy. This ended up being due to the fact that I was attempting to make a prediction based on the image every frame, or every 1/25th second. To aliviate the issue, I did some performance debugging and found that each segmentation + prediction took around .09s. To give the computer the time it needed (and then some), I updated the application to only make predictions every 6 frames, giving the tool ~0.25s to process. This aliviated my problem. With a more powerful computer, I likely could have processed a prediction once per frame, as that would only need a ~3x speed improvement. 

### Current State of the Art

Due to the vast potential applications for a robust noise-agonistic hand gesture recognition system, there has been lots of research on the topic. In a recent [review, Munir Oudah, Ali Al-Naji, and Javaan Chahl](https://www.mdpi.com/2313-433X/6/8/73/pdf) analyzed the current state of modern hand gesture recognition systems. The paper split the results of various researchers in categories based on the approach. I will highlight the methods that are similar to mind and the given accuracy bellow:

#### Appearance Based Detection
Appearance based detection of hands and gestures using features detected in 2D images. This appraoch is very similar to the one I took in doing this assignment, except they did not use deep models. The current best approach in this category was in a paper "Fang, Y.; Wang, K.; Cheng, J.; Lu, H. A real-time hand gesture recognition method." and was able to recognize 6 hand gestures at 93% accuracy using a Gaussian Model for skin color and segmentation, and a palm-finger configuration algorithm for classificaiton.

#### Motion Based Detection
Motion based detection uses a series of images taken within close proximity of one another to detect the hand and its features, which are then ran through a model for prediction (in what I believe to be something similar to the optical flow content). This method proved to provide beter results than apperance based recognition. In the paper "Pun, C.-M.; Zhu, H.-M.; Feng, W. Real-time hand gesture recognition using motion tracking.Int. J. Comput.Intell. Syst.2011,4, 277–286.", researchers were able to achieve an accuracy of 97.33% for 10 gestures using color and motion tracking segmentation with a histogram distribution model. Note that this method did not work on static images. 

#### Skeleton Based Recognition
Skeleton based recognition involves mapping a skeleton of the hand on top of a hand in the training dataset and from arbitrary images. The skeltal structure is then used as the features that are ran through a model for hand gesture recognition. In the paper "Devineau, G.; Moutarde, F.; Xi, W.; Yang, J. Deep learning for hand gesture recognition on skeletal data.In Proceedings of the 2018 13th IEEE International Conference on Automatic Face & Gesture Recognition(FG 2018)" researchers were able to classify 14 gestures with 91.28% accuracy using a CNN. Note that this method required a special camera that could generate skeleton feature data. 

#### Deep Learning Based Recognition 
The area where researchers have seen the most success in terms of hand gesture recognition is with Deep Learning based approaches. These methods provide results with accuracy many percentage points higher than the aforementioned methods. The current champion in this category is the paper "Alnaim, N.; Abbod, M.; Albar, A. Hand Gesture Recognition Using Convolutional Neural Network forPeople Who Have Experienced A Stroke." In this paper, the researchers were able to achieve an accuracy of 100% on the training set and 99% on the test set for 7 different hand gestures. They handled feature extraction using CNN techniques, and the model as a whole was an Adapted Deep Convolutional Neural Network custom built for this experiment . With this accuracy, the model was likely as good as if not better than a human doing the same task.

#### Depth Based Recognition
Researchers were able to get great results using 3D cameras for hand gesture recognition. All of these approaches require 3D cameras, or 2D cameras paired with depth sensors. The first dataset I mentioned above was actually a dataset from one of these depth-based recognition papers! The best paper in this areas was from the XBOX Kinect team who were able to achieve 96% accuracy using this method. 


