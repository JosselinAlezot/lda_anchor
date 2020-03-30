import pandas as pd
from framer import Framer


class GroupFramer(Framer):

    def create_frames(self):
        """
        Creates dataframes list containing every dataframe for every label
        in the dataset. Each row of a dataframe represents a group (users sharing the same label)
        and its text is the concatenation of its users' posts,
        i.e. each group of user is considered as a document
        """
        self.dataframes = {}

        for label in self.labels:
            self.dataframes[label] = pd.DataFrame(columns=['id','text','posts_counts','label'])
            self.dataframes[label].loc[label] = [label] + [''] + [0] + [label]

        for row in self.data:
            text = ''
            counts = 0
            for post in row['posts']:
                text = text + post['text']
                counts += 1
            self.dataframes[row['label']].loc[row['label'], 'text'] += text
            self.dataframes[row['label']].loc[row['label'], 'posts_counts'] += counts
