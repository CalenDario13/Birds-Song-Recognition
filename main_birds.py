from header import PATH_DIRECTORY, PATH_FILE
import os
os.chdir(PATH_DIRECTORY)
import time
from new_retriver import top_ten, retriver, process_manager
from preprocessing import Cleaner

import pyarrow as pa
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
    df['background']
    '''
    # Preprocessing
    cleaner = Cleaner(df)
    df = cleaner.generate_final_db()
    '''
    

    
    
        
    
    
       

        
                
        
                 
                 

