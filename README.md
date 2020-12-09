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

1. Visit the Kaggle website and download [the Hand Gesture Recognition Database](https://www.kaggle.com/gti-upm/leapgestrecog)

2. Move the downloaded dataset into the project: `mv ~/Downloads/leapGestRecog ./data`

## Implementation Details

See the project webpage: https://joeholt.github.io/HandGestureRecognizer/
