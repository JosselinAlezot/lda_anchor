import pandas as pd
from framer import Framer


class PostFramer(Framer):

    def create_frames(self):
        """
        Creates dataframes list containing every dataframe for every label
        in the dataset. Each row of a dataframe represents a post,
        i.e. each post is considered as a document, and user authorship is not considered.
        """
        self.dataframes = {}

        for label in self.labels:
            self.dataframes[label] = pd.DataFrame(columns=['id','text'])

        for row in self.data:
            print("tour de boucle")
            for post in row['posts']:
                row_id = row['id'] + '.' + post['date']
                self.dataframes[row['label']].loc[row_id] = [row_id] + [post['text']]
