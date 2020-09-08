import pandas as pd
import numpy as np
from collections import defaultdict

class Cleaner():
    
    def __init__(self, df):
        
        self.df = df
        
    
    def clean_type(self, row):
        
        # Define the columns domain:
        
        calls = set(['uncertain', 'song', 'subsong', 'call', 'alarm call', 'flight call', 'nocturnal flight call', 
             'begging call', 'drumming', 'duet', 'dawn song'])
        sex = set(['male', 'female', 'sex uncertain'])
        stage = set(['adult', 'juvenile', 'hatchling or nestling', 'life stage uncertain'])
        special = set(['aberrant', 'mimicry/imitation', 'bird in hand'])
        
        # Start selection:
        
        ca = []
        se = []
        st = []
        sp = []
        
        for tag in row:
            tag = tag.strip()
            if tag in calls and tag != 'uncertain':
                ca.append(tag)
            elif tag in sex and tag != 'sex uncertain':
                se.append(tag)
            elif tag in stage and tag != 'life stage uncertain':
                st.append(tag)
            elif tag in special:
                sp.append(tag)
        
        return ca, se, st, sp
    
    def transform_lst(self, lst):
        
        if len(lst) > 1:
            new = ', '.join(lst)
            return new
        elif len(lst) == 1:
            new = lst[0]
            return new
        elif len(lst) == 0:
            new = np.NaN
            return new
    
    def add_type_columns(self):
        
        tags = defaultdict(list)
        for row in self.df['type']:
            
            # Add the tag to the corresponding column lst:
            ca, se, st, sp = self.clean_type(row)
            
            # Add the tags of the row to the dictionary:
            ca = self.transform_lst(ca)
            tags['call'].append(ca)
            se = self.transform_lst(se)
            tags['sex'].append(se)
            st = self.transform_lst(st)
            tags['stage'].append(st)
            sp = self.transform_lst(sp)
            tags['special'].append(sp)
        
        type_df = pd.DataFrame.from_dict(tags)
        self.df = pd.concat([self.df.iloc[:,0:7], type_df, self.df.iloc[:,8:]], axis = 1)
        return self.df
        
        
        
        


