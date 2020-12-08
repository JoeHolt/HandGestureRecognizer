# controls functions to contorl computer (mac)
from subprocess import call

class ComputerInteraction():

    def __init__(self):
        self.predictions = []
        self.accs = []

    def add_prediction(self, preds, accs):
        """
        Adds predictions to tracker
        """
        self.predictions.insert(0, preds)
        self.accs.insert(0, accs)
        self.last_n_frames_class(4)

    def last_n_frames_class(self, n):
        """
        Returns the class of the last n frames if it was the same
        else returns none
        """
        last_class = self.predictions[0][0]
        for idx in range(1, min(len(self.predictions), n)):
            if self.predictions[idx][0] != last_class:
                return None
        print("Gesture Recognized for {} frames: {}".format(n, last_class))
        sendNotification(last_class, self.accs[0][0])
        return last_class


# Action methods
def setVolume(percent):
    """
    Sets sound to a given level
    """
    command = "osascript -e 'set volume output volume {}'".format(percent)
    call([command], shell=True)

def sendNotification(disc, title):
    """
    Sends notification
    """
    header = "Hand Gesture Recognizer"
    text = "{}: {}".format(disc, title)
    command = """
    osascript -e 'display notification "{}" with title "{}"'
    """.format(text, header)
    call([command], shell=True)
