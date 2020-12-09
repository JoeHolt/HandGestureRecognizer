# controls functions to contorl computer (mac)
from subprocess import call

class ComputerInteraction():

    def __init__(self):
        self.predictions = []
        self.accs = []
        self.action_threshold = 6
        self.action_handlers = {
            'ok': self.handle_ok,
            'l': self.handle_fist
        }

    def add_prediction(self, preds, accs):
        """
        Adds predictions to tracker
        """
        self.predictions.insert(0, preds)
        self.accs.insert(0, accs)
        self.last_n_frames_class(8)

    def last_n_frames_class(self, n):
        """
        Returns the class of the last n frames if it was the same
        else returns none
        """
        last_class = self.predictions[0][0]
        for idx in range(1, min(len(self.predictions), n)):
            if self.predictions[idx][0] != last_class:
                return None
            
        if last_class in self.action_handlers.keys():
            print("Gesture Recognized for {} frames: {}".format(n, last_class))
            acc = self.accs[0][0]
            func = self.action_handlers[last_class]
            func(acc)

        return last_class

    def handle_ok(self, acc):
        self.sendNotification('OK', acc)
        self.setVolume(75)

    def handle_fist(self, acc):
        self.sendNotification('L', acc)
        self.setVolume(0)

    def setVolume(self, percent):
        """
        Sets sound to a given level
        """
        command = "osascript -e 'set volume output volume {}'".format(percent)
        call([command], shell=True)

    def sendNotification(self, class_str, accuracy):
        """
        Sends notification
        """
        header = "Hand Gesture Recognizer"
        text = "{}: {}".format(class_str, round(accuracy*100,3))
        command = """
        osascript -e 'display notification "{}" with title "{}"'
        """.format(text, header)
        call([command], shell=True)
