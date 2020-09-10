import os
os.chdir('/Users/dario/Google Drive/DS/First Year - Secon Semester/SL/final_project/')
import time

from header import PATH_DIRECTORY, PATH_FILE
from new_retriver import top_ten, retriver, process_manager
from preprocessing import Cleaner, Audio_Processing, multi_df

import pyarrow.parquet as pq

if __name__ == '__main__':
        
    # Declear variables:
        
    BASE = 'https://www.xeno-canto.org'

    TIME_LEN = 5
    QUALITY_RATE = 22050
    FRAME_LEN = 1024
    HOP_LEN = 512

    # Get birds with more  recordings:
    if os.path.exists(PATH_FILE + '.txt'):
        with open(PATH_FILE + '.txt', 'r') as f:
            birds_link = f.readlines()
            birds_link = [link[:-1] for link in birds_link]
    else:
        top = top_ten(BASE, PATH_FILE)   
        birds_link = top.top_birds()

    # Get Data:

    retr = retriver(BASE, PATH_FILE, PATH_DIRECTORY, TIME_LEN, QUALITY_RATE, FRAME_LEN, HOP_LEN)
    process_manager(retr.get_data, birds_link[:5])
    time.sleep(60)
    process_manager(retr.get_data, birds_link[5:])

    # Create only one prquet:
    
    retr.merge_parquets()

    # Import table and convert in pandas
    print('Start loading df\n...')
    table = pq.read_table('birds.parquet')
    df = table.to_pandas()
    print('Done!')
    
    
    # Preprocessing

    cleaner = Cleaner(df)
    df = cleaner.generate_final_db()
    
    process = Audio_Processing(df, QUALITY_RATE, HOP_LEN)

    mel = multi_df(process.transform_df, df)
 
    df.columns[:18]
                 
                 

