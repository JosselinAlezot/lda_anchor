import pandas as pd
from framer import Framer


class UserFramer(Framer):

    def create_frames(self):
        """
        Creates dataframes list containing every dataframe for every label
        in the dataset. Each row of a dataframe represents a user
        and its text is the concatenation of the user's posts,
        i.e. each user is considered as a document
        """
        self.dataframes = {}

        for label in self.labels:
            self.dataframes[label] = pd.DataFrame(columns=['id','text','posts_counts','label'])

        for row in self.data:
            text = ''
            counts = 0
            for post in row['posts']:
                text = text + post['text']
                counts += 1
            self.dataframes[row['label']].loc[row['id']] = [row['id']] + [text] + [counts] + [row['label']]
