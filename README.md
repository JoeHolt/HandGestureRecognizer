# HandGestureRecognition
Project for CS 639 (Computer Vision) that allows you to use hand gestures to control your computer. 

## Installation and Usage

Create a new conda enviornment with required packages:

```bash
conda env create --name envname --file=environments.yml
```

Run the program from the root directory:

```bash
python ./src/main.py
```

## Loading Training Data

If you would like to run the notebooks and/or continue model development, you will need to download the training data separately and place it in the `./data/` directory:

1. Visit the Kaggle website and download [the leapgestercog dataset](https://www.kaggle.com/gti-upm/leapgestrecog)

2. Move the downloaded dataset into the project: `mv ~/Downloads/leapGestRecog ./data`

## Implementation Details

At the highest level, this application consists of a web cam run loop where each iteration of the loop, the web cam pull a new image frame which is then processed for gesture recogition. The image pipeline is described bellow.

### Segmenting the Hand from Image Frame

To segment the hand from the background image, I make use of a simply binary mask between the initial frame and the current frame. To create this mask, I grab a webcam frame, convert it to a black and white photo, then subtract the initial frame from it using OpenCV.

In this set up, I expect that the initial frame (ie the first frame that is read when launching the program) is an empty room with no one in it. Furthermore, I expect that the webcam does not move while this program is active.

This mask is able to segment the image because (ideally) the only thing in the webcams POV that has changed is the addition of a hand with a gesture on it.
