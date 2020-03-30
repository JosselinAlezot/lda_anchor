from erisk2020data.handler import Task
import pandas as pd
from abc import ABC, abstractmethod


class Framer(ABC):
    
    def __init__(self, path, labels):
        """
        Constructor for the framer. Creates listed data 
        from the data given by the path 
        :param path: Path to the data to be framed
        :param labels: Array of the labels existing in the data to be framed
        """
        task = Task(path)
        unlisted_data = task.get_data()
        self.data = list(unlisted_data)
        self.labels = labels

    @abstractmethod
    def create_frames(self):
        """
        Abstract method to be implemented by subclasses
        """
        pass

    def get_frame(self, label):
        """
        Gets frame by label in dataframes list
        :param label: Label of the wanted frame
        :return: Frame for the given label
        """
        return self.dataframes[label]

    def get_concat_frames(self):
        """
        Gets frame by label in dataframes list
        :param label: Label of the wanted frame
        :return: Frame for the given label
        """
        return pd.concat(self.dataframes)

    def get_bigram_ready_frame(self):
        text = ' '.join(pd.concat(self.dataframes)['text'].tolist())
        bigram_frame = pd.DataFrame({'text':[text]})
        return bigram_frame








