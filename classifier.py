import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics  import f1_score


class Classifier:

    def __init__(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)

        kf = KFold(5, shuffle=True, random_state=42)
        cv_lr_f1, cv_lrsgd_f1, cv_svcsgd_f1,  = [], [], []

        for train_ind, val_ind in kf.split(self.X, self.y):
            # Assign CV IDX
            X_train, y_train = self.X[train_ind], self.y[train_ind]
            X_val, y_val = self.X[val_ind], self.y[val_ind]

            # Scale Data
            scaler = StandardScaler()
            X_train_scale = scaler.fit_transform(X_train)
            X_val_scale = scaler.transform(X_val)

            # Logisitic Regression
            lr = LogisticRegression(
                class_weight= 'balanced',
                solver='newton-cg',
                fit_intercept=True
            ).fit(X_train_scale, y_train)

            y_pred = lr.predict(X_val_scale)
            cv_lr_f1.append(f1_score(y_val, y_pred, average='binary', pos_label='1'))

            # Logistic Regression SGD
            sgd = SGDClassifier(
                max_iter=1000,
                tol=1e-3,
                loss='log',
                class_weight='balanced'
            ).fit(X_train_scale, y_train)

            y_pred = sgd.predict(X_val_scale)
            cv_lrsgd_f1.append(f1_score(y_val, y_pred, average='binary', pos_label='1'))

            # SGD Modified Huber
            sgd_huber = SGDClassifier(
                max_iter=1000,
                tol=1e-3,
                alpha=20,
                loss='modified_huber',
                class_weight='balanced'
            ).fit(X_train_scale, y_train)

            y_pred = sgd_huber.predict(X_val_scale)
            cv_svcsgd_f1.append(f1_score(y_val, y_pred, average='binary', pos_label='1'))

        print(f'Logistic Regression Val f1: {np.mean(cv_lr_f1):.3f} +- {np.std(cv_lr_f1):.3f}')
        print(f'Logisitic Regression SGD Val f1: {np.mean(cv_lrsgd_f1):.3f} +- {np.std(cv_lrsgd_f1):.3f}')
        print(f'SVM Huber Val f1: {np.mean(cv_svcsgd_f1):.3f} +- {np.std(cv_svcsgd_f1):.3f}')

