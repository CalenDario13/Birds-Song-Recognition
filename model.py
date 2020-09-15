import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score

class Classifier():
    
    def __init__(self, df):
        
        self.id_class = {k : v for v , k in enumerate(df['common_name'].unique())}
        self.labels = df['common_name'].apply(lambda x: self.id_class[x])
        self.df = df
    
    def fill_na(self, train, test, strategy):
        
        # Working on categorical variables:
        
        imputer = SimpleImputer(missing_values = np.NaN, strategy  = strategy)
        imputer.fit(train)
        
        train_imp = imputer.transform(train)  
        test_imp = imputer.transform(test)  
        
        train = pd.DataFrame(train_imp, columns = train.columns)
        test = pd.DataFrame(test_imp, columns = test.columns)
        
        return train, test
    
    def scaling(self, train, test):
        
        scaler = StandardScaler()
        scaler.fit(train)

        scaled_train = scaler.transform(train)
        scaled_test = scaler.transform(test)

        train = pd.DataFrame(scaled_train, columns = train.columns)
        test = pd.DataFrame(scaled_test, columns = test.columns)
        
        return train, test
    
    def prepare_df(self):
        
        # Split df:
        print('- Split Dataframe')
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.df, self.labels, test_size = 0.20, random_state = 100)
        
        # Fill na:
        print('- Impute missing values')
        dummy_train = self.X_train[['country','gio_not', 'season', 'call', 'sex', 'stage', 'special', 'elevation']]
        dummy_test = self.X_test[['country','gio_not', 'season', 'call', 'sex', 'stage', 'special', 'elevation']]
        
        dummy_train, dummy_test = self.fill_na(dummy_train, dummy_test, strategy = 'most_frequent')
        
        other_train = self.X_train[['latitude', 'longitude', 'centroids']]  
        other_test = self.X_test[['latitude', 'longitude', 'centroids']] 
        self.X_train.drop(columns = ['centroids'], inplace = True)
        self.X_test.drop(columns = ['centroids'], inplace = True)
        
        other_train, other_test = self.fill_na(other_train, other_test, strategy = 'mean')
         
        # Scale variables:
        
        print('- Scale variables')
        scaled_train, scaled_test = self.scaling(other_train, other_test)
        
        # Transform in dummies:
        print('- Encode dummies')  
        
        idx_split = len(dummy_train)
        complete_dummy = pd.concat([dummy_train, dummy_test], axis = 0, ignore_index=True)
        
        dummy = pd.get_dummies(complete_dummy, drop_first = True)
       
        dummy_train = dummy.iloc[:idx_split, :]
        dummy_test = dummy.iloc[idx_split:, :]
        
        # Combine everything:
        
        print('- Generate final DataFrames')
                        
        self.X_train = np.concatenate((dummy_train.values, scaled_train.values, self.X_train.iloc[:, 17:].values), axis=1)
        self.X_test = np.concatenate((dummy_test.values, scaled_test.values, self.X_test.iloc[:, 17:].values), axis=1)
      
    def new_evaluation_score(self, classifier, score_weights=np.linspace(0,10,10)/10):

        '''The new score takes into account the position in the sorted predicted list
        of the true label. According to its position it gives a score in [0,1], where
        0 is when the true label is in the last position and 1 when in the first.'''

        pred_probabilities = classifier.predict_proba(self.X_test)
        new_score = 0
        target = self.y_test.values.astype(int)
        for i in range(len(pred_probabilities)):
            current_target = target[i]
            current_prediction = np.argsort(pred_probabilities[i])
            new_score += score_weights[np.where(current_prediction == current_target)[0][0]]
        new_score = new_score/target.shape[0]
        return new_score

    def class_score_df(self, classifier, id_class_map, real_names=True):

        '''This function returns a dataframe where each column is associated to a datapoint
        with name of the column corresponding to the true label. Each column represents the
        sorted classes according to the probability output of the classifier.'''

        pred_probabilities = classifier.predict_proba(self.X_test.values)
        index = np.array(list(id_class_map.keys()))
        column_names = index[self.y_test.values.astype(int)]
        class_score = pd.DataFrame()
        for row in pred_probabilities:
            if real_names:
                class_score = pd.concat([class_score,pd.Series(index[np.argsort(row)[::-1]])], axis=1)
            else:
               class_score = pd.concat([class_score,pd.Series(np.argsort(row)[::-1])], axis=1)
        if real_names:
            class_score.columns = column_names
        else:
            class_score.columns = self.y_test.values.astype(int)
        return class_score
        
        clf = SVC(C = 8)
        clf.fit(self.X_train, self.y_train)
        s = clf.score(self.X_test, self.y_test)
        print('Test score is: {}'.format(s))

    def test_model(self):

        clf1 = SVC(C=24, probability=True)
        clf2 = LogisticRegression(C=0.45, max_iter=2000)
        clf3 = RandomForestClassifier(n_estimators=4000)

        vote_clf = VotingClassifier(estimators=[('SVC', clf1), ('Logistic', clf2),
                                                ('RandomForest', clf3)], voting='soft')

        vote_clf.fit(self.X_train, self.y_train)
        #self.class_score_df(vote_clf, self.id_class,real_names=False).to_csv('predictions_classes_sorted.csv', index=False)
        #self.y_test.to_csv('test_target.csv')
        return vote_clf.score(self.X_test, self.y_test)

    def evaluate_model(self):
        
        self.prepare_df()
        print('- Cross Validation Evaluation')

        clf1 = SVC(C=24, probability=True)
        clf2 = LogisticRegression(C=0.45, max_iter=2000)
        clf3 = RandomForestClassifier(n_estimators=5000)

        vote_clf = VotingClassifier(estimators=[('SVC', clf1), ('Logistic', clf2), ('RandomForest', clf3)],
                                    voting='soft')
        score = cross_val_score(vote_clf, self.X_train, self.y_train, n_jobs = -1, scoring = 'accuracy', cv = 10)
        vote_clf.fit(self.X_train, self.y_train)
        print('Our Score:', self.new_evaluation_score(vote_clf))
        print('Cross-Validation Score:', np.mean(score))
        #self.class_score_df(vote_clf, self.id_class).to_csv('predictions.csv', index=False)
        print('Test Score:', self.test_model())
    
 