import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.decomposition import KernelPCA  

from sklearn.preprocessing import scale
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

class Classifier():
    
    def __init__(self, df):
        
        id_class = {k : v for v , k in enumerate(df['common_name'].unique())}
        self.labels = df['common_name'].apply(lambda x: id_class[x])
        self.df = df
    
    def binning_elev(self, el):
  
        if el <= 240:
            binn = 'bassa'
        elif 240 < el <= 500:
            binn = 'media'
        else:
            binn = 'alta'
            
        return binn
    
    def prepare_df(self, pca = False):
        
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
       
        other_df = scale(other_df)
        other_df = pd.DataFrame(other_df, columns = ['latitude', 'longitude'])
        
        # Combine everything:
 
        self.df = pd.concat([dummy_df, other_df, self.df.iloc[:,17:]], axis = 1)
        
        if pca:
            
            kpca_transform = KernelPCA(n_components = 80, kernel='cosine')
            kpca_df = kpca_transform.fit_transform(self.df)
            
            '''
            explained_variance = np.var(kp, axis=0)
            explained_variance_ratio = explained_variance / np.sum(explained_variance)
            
            plt.plot(np.cumsum(explained_variance_ratio))
            plt.xlabel('number of components')
            plt.ylabel('cumulative explained variance')
            '''
            
            # Split df:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(kpca_df, self.labels, 
                                                                test_size = 0.20, random_state = 100)
        
        else:
            
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.df, self.labels, 
                                                                test_size = 0.20, random_state = 100)
            
    def test_model(self, best_sc):
        
        clf = SVC(C = best_sc)
        clf.fit(self.X_train, self.y_train)
        s = clf.score(self.X_test, self.y_test)
        print('Test score is: {}'.format(s))
    
    def evaluate_model(self):
        
        self.prepare_df()
        #8
        C_vals = [5, 8, 10, 50]
        scores = []
        for c in C_vals:
            clf = SVC(C = c)
            score = cross_val_score(clf, self.X_train, self.y_train, 
                                    n_jobs = -1, scoring = 'accuracy', cv = 10)
            scores.append((np.mean(score), c))
            print(np.mean(score), c)
      
        if len(scores) == 1:
            best = scores[0][1]
        else:
            best = sorted(scores, key = lambda tpl: tpl[0], reverse = True)[0][1]
        
        self.test_model(best)   
        
    

