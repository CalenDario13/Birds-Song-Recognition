import numpy as np
import pandas as pd

from sklearn.utils import shuffle
from sklearn.preprocessing import scale
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

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
    
    def binning_elev(self, el):
  
        if el <= 240:
            binn = 'bassa'
        elif 240 < el <= 500:
            binn = 'media'
        else:
            binn = 'alta'
            
        return binn
    
    def prepare_df(self):
        
        # Fill NaN:
        dummy_df = self.df[['country','gio_not', 'season', 'call', 'sex', 'stage', 'special']]
        imp_mode = SimpleImputer(missing_values = np.NaN, strategy  ='most_frequent')
        dummy_df = imp_mode.fit_transform(dummy_df)   
        dummy_df = pd.DataFrame(dummy_df, columns = ['country','gio_not', 'season', 'call', 'sex', 'stage', 'speciale'])
        
        other_df = self.df[['latitude', 'longitude', 'elevetaion']]        
        imp_mean = SimpleImputer(missing_values=np.nan, strategy = 'mean')
        other_df = imp_mean.fit_transform(other_df)
        other_df = pd.DataFrame(other_df, columns = ['latitude', 'longitude', 'elevetaion'])
        
        # Transform elevation in dummy:
        
        dummy_df['elevation'] = other_df.elevetaion.apply(lambda x: self.binning_elev(x))
        other_df.drop(columns=['elevetaion'], inplace = True)
        
        # Transform dummies:
        
        dummy_df = pd.get_dummies(dummy_df, drop_first = True)
        
        # Scaling:
       
        scale_freq = scale(self.df.iloc[:,17: -1], axis = 1)
        freq_df = pd.DataFrame(scale_freq)
        other_df = scale(other_df)
        other_df = pd.DataFrame(other_df, columns = ['latitude', 'longitude'])
        
        # Combine everything:
 
        self.df = pd.concat([dummy_df, other_df, freq_df, self.df[['centroids']]], axis = 1)
        
        # Split df:
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.df, self.labels, 
                                                            test_size = 0.20, random_state = 100)
        
        
    def new_evaluation_score(self, classifier, score_weights=np.linspace(0,10,10)/10):
        
        '''The new score takes into account the position in the sorted predicted list
        of the true label. According to its position it gives a score in [0,1], where 
        0 is when the true label is in the last position and 1 when in the first.'''
        
        pred_probabilities = classifier.predict_proba(self.X_test.values)
        new_score = 0
        target = self.y_test.values.astype(int)
        for i in range(len(pred_probabilities)):
            current_target = target[i]
            current_prediction = np.argsort(pred_probabilities[i])
            new_score += score_weights[np.where(current_prediction == current_target)[0][0]]
        new_score = new_score/target.shape[0]
        return new_score
        
    def class_score_df(self, classifier, id_class_map):
        
        '''This function returns a dataframe where each column is associated to a datapoint
        with name of the column corresponding to the true label. Each column represents the 
        sorted classes according to the probability output of the classifier.'''
        
        pred_probabilities = classifier.predict_proba(self.X_train.values)
        index = np.array(list(id_class_map.keys()))
        column_names = index[self.y_train.values.astype(int)]
        class_score = pd.DataFrame()
        for row in pred_probabilities:
            class_score = pd.concat([class_score,pd.Series(index[np.argsort(row)[::-1]])], axis=1)
        class_score.columns = column_names
        return class_score
        
    def evaluate_model(self):
        
        self.prepare_df()
        
        
        C_vals = [0.2,0.25,0.3]
        
        for c in C_vals:
            clf = LogisticRegression(C = c, max_iter=1000)
            score = cross_val_score(clf, self.X_train, self.y_train, 
                                    n_jobs = -1, scoring = 'accuracy', cv = 10)
            print(np.mean(score), c)
            
        '''
        
        clf1 = SVC(C=20, probability=True)
        clf2 = LogisticRegression(C=0.3, max_iter=1000)
        clf3 = RandomForestClassifier(n_estimators=4000)
        
        vote_clf = VotingClassifier(estimators=[('SVC', clf1), ('Logistic', clf2), 
                                                ('RandomForest', clf3)], voting='soft')
        #score = cross_val_score(vote_clf, self.X_train, self.y_train, 
                                #  n_jobs = -1, scoring = 'accuracy', cv = 10)
        vote_clf.fit(self.X_train, self.y_train)
        print(self.new_evaluation_score(vote_clf))
        #print(np.mean(score))
        #self.class_score_df(vote_clf, self.id_class).to_csv('predictions.csv', index=False)
        '''
    def test_model(self):
        self.prepare_df()
        clf1 = SVC(C=20, probability=True)
        clf2 = LogisticRegression(C=0.3, max_iter=1000)
        clf3 = RandomForestClassifier(n_estimators=4000)
        
        vote_clf = VotingClassifier(estimators=[('SVC', clf1), ('Logistic', clf2), 
                                                ('RandomForest', clf3)], voting='soft')
        
        vote_clf.fit(self.X_train, self.y_train)
        print(vote_clf.score(self.X_test, self.y_test))
 