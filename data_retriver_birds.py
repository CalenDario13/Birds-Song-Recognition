import pandas as pd
import numpy as np
import heapq
import os.path
import warnings
from tqdm import tqdm
import re

import threading

import requests
from requests_toolbelt import sessions
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from bs4 import BeautifulSoup as bs

from io import BytesIO
from pydub import AudioSegment
from scipy.signal import find_peaks
import librosa


def get_table(url):
    
    session_0 = requests.Session() 
    retry = Retry(total = 10000, backoff_factor = 0.5)
    adapter = HTTPAdapter(max_retries = retry)
    session_0.mount('http://', adapter)
    session_0.mount('https://', adapter)
    
    r = session_0.get(url)
    soup = bs(r.content, "html5lib")
    
    #Find the table
    table = soup.find('table', {'class': 'results'})
    
    # Find columns name:
    cols = table.find('thead')
    cols = cols.find_all('th')
    cols = [re.sub(r'\s{2,}', ' ', col.text) for col in cols]
    cols = ['Unamed'] + cols[1:]

    # Find elements:
        
    body = table.find('tbody')
    rows = table.find_all('tr')

    to_df = []
    for row in rows[1:]:
        els = row.find_all('td')
        attr = []
        for el in els:
            attr.append(re.sub(r'\s{2,}', ' ', el.text.strip()))
        to_df.append(attr)
        
    df = pd.DataFrame(to_df, columns = cols)

    #Get scores:
    rat_soup = body.find_all('div', {'class': 'rating'})

    ratings = []
    for rate in rat_soup:
        
        try:
            selected = rate.find('li', {'class':'selected'})
            ratings.append(selected.text)
        except:
            ratings.append(None)
    
    df.Actions = ratings
        
    # Convert the length of the audio in seconds:
        
    df['Length'] = df['Length'].apply(lambda x:  sum([a * b for a,b in zip([60, 1], map(int, x.split(':')))]) 
                                            if len(x.split(':')) == 2
                                            else sum([a * b for a,b in zip([3600, 60, 1], map(int, x.split(':')))]))

        
    return df

def cut_longer_audio (wave, time_len, quality_rate, frame_len, hop_len, sample_rate):
    
    rms_wave = librosa.feature.rms(wave, frame_length = frame_len, hop_length = hop_len)[0]
    peak_pos = find_peaks(rms_wave)[0]
    peak_tr = np.quantile(rms_wave[peak_pos], q = 0.95)
    random_peak = np.random.choice(a = np.where(rms_wave > peak_tr)[0])
    del rms_wave
    
    division_peak = random_peak * hop_len
    lower_cut = division_peak - time_len/2 * sample_rate
    upper_cut = division_peak + time_len/2 * sample_rate

    if lower_cut < 0:
        upper_cut += abs(lower_cut)
        lower_cut = 0
    elif upper_cut > len(wave):
        lower_cut -= abs(upper_cut - len(wave))
        upper_cut = len(wave)-1
    
    selected_audio = wave[int(lower_cut) : int(upper_cut)]
    
    
    if selected_audio.shape[0] < quality_rate * time_len:
        selected_audio = np.concatenate((selected_audio, np.zeros(quality_rate*time_len-selected_audio.shape[0])))
    
    return selected_audio

def get_song(url, audio_lst, time_len, quality_rate, frame_len, hop_len):
    
    session_1 = requests.Session() 
    retry = Retry(total = 10000, backoff_factor = 0.5)
    adapter = HTTPAdapter(max_retries = retry)
    session_1.mount('http://', adapter)
    session_1.mount('https://', adapter)
    
    try:
        
        response = session_1.get(url)
        data = response.content
        
        sound = AudioSegment.from_file(BytesIO(data)).set_channels(1).set_frame_rate(quality_rate)
        wave = np.array(sound.get_array_of_samples()).astype(float)
        sample_rate = sound.frame_rate
    
        if wave.shape[0] < quality_rate * time_len:
        
            wave = np.concatenate((wave, np.full(quality_rate * time_len-wave.shape[0], np.nan)))
        
        else:
            
            wave = cut_longer_audio(wave, time_len, quality_rate, frame_len, hop_len, sample_rate)
        
        xcid = ''.join(['XC', re.search(r'\d+', url)[0]])
        to_df = [xcid] + wave.tolist()
        audio_lst.append(to_df)
    
    except:
        
        audio_lst.append([np.nan for _ in range(time_len * quality_rate)])
    
def thread_manager(url_lsts, audio_lst, time_len, quality_rate, frame_len, hop_len):
    
    threads = []
    for url_download in url_lsts:
        
        t = threading.Thread(target = get_song, args = (url_download, audio_lst, time_len, quality_rate, frame_len, hop_len,) )
        t.start()
        threads.append(t)
        
    for process in threads:
        process.join()

def get_data(path_csv, time_len = 5, quality_rate = 22050, frame_len = 1024, hop_len = 512):
     
    # Find last page to parse:
     
    main_pg = requests.get('https://www.xeno-canto.org/explore?query=%20cnt%3A%22%3DItaly%22')
    main_soup = bs(main_pg.content, "html.parser")
    
    bar = main_soup.find('nav', {'class': 'results-pages'})
    last_pg = int(bar.find_all('li')[-2].text)
     
    # Start parsing:
    
    for n in tqdm(range(1, last_pg)): 

        if n % last_pg == 0:
            print('Last page has been processed')
           
           
        # Get the table in the website:
        
        base = 'https://www.xeno-canto.org/explore?query=+cnt%3A%22%3DItaly%22&pg='
        main_url = ''.join([base, str(n)])
        try:
            chunk = get_table(main_url)
        except:
            continue
        
        # Remove not needed rows and download audio files:
            
        idx_remove = []
        common = []
        scientific = []
        urls = []
        for row in chunk.itertuples():
            
            try:
                seen = re.search(r'(?<=bird-seen:)\w+', row.Remarks)[0]
            except:
                seen = 'no'
                
            score = row.Actions
            uncertain = bool(re.search(r'\[also\]', row.Remarks))
            
            if seen == 'yes' and (row._2 != '(?) Identity unknown' or row._2 != 'Soundscape') and (score == 'A' or score == 'B' or not score) and row.Length < 120 and not uncertain:
                
                # Split common name from scientific:
                
                try:
                    result = re.search(r'(.*\s)(\(.*\))', row._2)
                    common.append(result[1].strip())
                    scientific.append(result[2][1:-1])
                except:
                    common.append(row._2)
                    scientific.append(row._2)
                
                # Save song urls:
                
                xc_id = row._13
                reg = re.search(r'\d+', xc_id)
                xc_num = reg[0]
                url_sound = ''.join(['https://www.xeno-canto.org/', xc_num , '/download'])
                urls.append(url_sound)
            
            else:
                
                idx_remove.append(row.Index) 
        
        # Download audio and get spectrogram:
         
        audio_lst = []
        thread_manager(urls, audio_lst, 5, 22050, 1024, 512)
        audio_df = pd.DataFrame(audio_lst, columns = ['id'] +['wf_' + str(i) for i in range(time_len * quality_rate)])
       
        
        # Adjust DF:
        
        chunk.drop(idx_remove, axis = 0, inplace = True)
        chunk.drop(['Unamed','Common name / Scientific', 'Country', 'Length', 'Recordist', 'Remarks', 'Actions'], 
                   axis = 1, inplace = True)
        chunk['common_name'] = common
        chunk['scientific_name'] = scientific
        
        chunk.columns = ['date', 'time', 'location', 'elevetaion', 'type', 'id', 'common_name', 'scientific_name']
        chunk = chunk[['id', 'common_name', 'scientific_name', 'date', 'time', 'location', 'elevetaion', 'type']]
        
        chunk = chunk.merge(audio_df, how = 'inner', on = 'id')
        
        # Save DF:
        
        if not os.path.exists(path_csv):     
            chunk.to_csv(path_csv, mode = 'w', header = True, index = False)
        else:
            chunk.to_csv(path_csv, mode = 'a', header = False, index = False)
         
        
    

    
