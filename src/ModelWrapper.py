# Wraps a model to get gesture predection
import torch as torch
import os
from ImageProcessing import process_ResNetV1

class GestureRecognizer:

    def __init__(self, model='default'):
        """
        Sets up the class and loads the given model
        """
        os.chdir('/Users/joeholt/Documents/College Local/Current/CS 639/proj/src')
        # can use multiple different clssified
        if model == 'default':
            model, classes, process = self._loadResNetV1()
            self.model = model
            self.classes = classes
            self.process = process

    def _loadResNetV1(self):
        """
        Loads resnet model trained earlier and the class
        names corresponding to each output index
        """
        classes = ['c', 'down', 'fist', 'index', 'l', 'ok', 'palm', 'thumb']
        model = torch.load('./models/resenet-v0-25ep.pth')
        model.eval()
        return model, classes, process_ResNetV1

    def predict(self, img):
        """
        Processes image, runs image through model and then
        returns prediction of image class
        """
        # process and run through model
        processed = self.process(img)
        result = self.model(processed)
        # get accuracy as probabilities
        softmax = torch.nn.Softmax(dim=1)
        accs = softmax(result).tolist()[0]
        # sort based on accuracy
        sorted_classes = reversed([c for _,c in sorted(zip(accs, self.classes))])
        sorted_accs = reversed(sorted(accs))

        return list(sorted_classes), list(sorted_accs)
