import pyarrow as pa
import pyarrow.parquet as pq

import os
PATH_DIRECTORY = '/Users/dario/Google Drive/DS/First Year - Secon Semester/SL/final_project/'
os.chdir(PATH_DIRECTORY)
from new_retriver import top_ten, retriver, process_manager

if __name__ == '__main__':
        
    # Declear variables:
        
    PATH_FILE = '/Users/dario/Google Drive/DS/First Year - Secon Semester/SL/final_project/birds'
    
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
    process_manager(retr.get_data, birds_link[5:])
    
    # Create only one prquet:
    print('I am creating the table')
    retr.merge_parquets()
    print('Done!')
    
    # Import table and convert in pandas
    table = pq.read_table('birds.parquet')
    df = table.to_pandas()
   
