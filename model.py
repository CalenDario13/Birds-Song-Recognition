import numpy as np
import pandas as pd

from sklearn.utils import shuffle
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
 
    def prepare_df(self):
        
        # Fill NaN:
        dummy_df = self.df[['country', 'gio_not', 'season']]
        imp_mode = SimpleImputer(missing_values = np.NaN, strategy  ='most_frequent')
        dummy_df = imp_mode.fit_transform(dummy_df)   
        dummy_df = pd.DataFrame(dummy_df, columns = ['country', 'gio_not', 'season'])
        
        other_df = self.df[['latitude', 'longitude', 'elevetaion']]        
        imp_mean = SimpleImputer(missing_values=np.nan, strategy = 'mean')
        other_df = imp_mean.fit_transform(other_df)
        other_df = pd.DataFrame(other_df, columns = ['latitude', 'longitude', 'elevetaion'])
        
        # Transform elevation in dummy:
        
        dummy_df['elevation'] = other_df.elevetaion.apply(lambda x: self.binning_elev(x))
        other_df.drop(columns=['elevetaion'], inplace = True)
        
        # Transform dummies:
        
        dummy_df = pd.get_dummies(dummy_df, drop_first = True)
        
        # Scale bins:
            
        scale_freq = scale(self.df.iloc[:,17: -1], axis = 1)
        freq_df = pd.DataFrame(scale_freq)
        
        # Combine everything:
 
        self.df = pd.concat([dummy_df, freq_df, self.df[['centroids']]], axis = 1)
        
        # Split df:
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.df, self.labels, 
                                                            test_size = 0.20, random_state = 100)
        
        
    def evaluate_model(self):
        
        self.prepare_df()
        
        C_vals = [14]

        for c in C_vals:
            clf = SVC(C = c)
            score = cross_val_score(clf, self.X_train, self.y_train, 
                                    n_jobs = -1, scoring = 'accuracy', cv = 10)
            print(np.mean(score))
    
       