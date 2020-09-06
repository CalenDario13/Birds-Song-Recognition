import pandas as pd
import numpy as np
import heapq
import os.path
import warnings
from tqdm import tqdm
import re

import threading
import multiprocessing as mp

import pyarrow as pa
import pyarrow.parquet as pq
from glob import glob

import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from bs4 import BeautifulSoup as bs

from io import BytesIO
from pydub import AudioSegment
from scipy.signal import find_peaks
import librosa

def thread_manager(function):
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=function, args=[item for item in args], kwargs=kwargs)
        thread.start()
        return thread
    return wrapper

def process_manager(function, lst):
    with mp.Pool(5) as p:
        list(tqdm(p.imap(function, lst), total = len(lst)))


class top_ten():
    
    def __init__(self, base, path):
        
        self.base_url = base
        self.path = path
        
    def establish_connection(self, link_path):
        
        session = requests.Session() 
        retry = Retry(total = 10000, backoff_factor = 0.5)
        adapter = HTTPAdapter(max_retries = retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        
        url = ''.join([self.base_url, link_path])
        r = session.get(url)
        soup = bs(r.content, 'html.parser')
        
        return soup
    
    def get_info(self, soup):
        
        name = soup.find('a').text.strip()
        score = int(soup.find('span', {'class': 'recording-count'}).text.strip())
        link_path = soup.find('a')['href']
        
        return name, score, link_path
    
    def get_top_n(self, n, sub_soup):
        
        rnk = []
        heapq.heapify(rnk)
        link_dic = {}
        for soup_el in sub_soup:
            
            try:
                name, score, link_path = self.get_info(soup_el)    
                heapq.heappush(rnk, (score, name))
                link_dic[name] = link_path   
            except:
                continue
        
        url_top = [link_dic[tupl[1]] for tupl in heapq.nlargest(n, rnk)]
        
        if n == 1: 
            return url_top[0]
        else:
            return url_top
        
    def top_taxonomy(self):
        
        path_url = '/explore/taxonomy'
        soup = self.establish_connection(path_url)
        taxonomies = soup.find_all('li')
        top_10 = self.get_top_n(10, taxonomies)
        
        return top_10
        
    def top_species(self):
        
        link_lst = self.top_taxonomy()
        
        top_species = []
        for link_path in link_lst:
            soup = self.establish_connection(link_path)
            sub_soup = soup.find('a', {'href': link_path}).next_sibling            
            species = sub_soup.find_all('li')
            top = self.get_top_n(1, species)
            top_species.append(top)
         
        return top_species

    def top_sub_species(self):
        
        link_lst = self.top_species()
        
        top_sub_species = []
        for link_path in link_lst:
            soup = self.establish_connection(link_path)
            sub_soup = soup.find(string = re.compile('\s\(.*\)')).next_sibling            
            sub_species = sub_soup.find_all('li')
            top = self.get_top_n(1, sub_species)
            top_sub_species.append(top)
         
        return top_sub_species
    
    def top_birds(self):
        
        link_lst = self.top_sub_species()
        
        top_birds = []
        for link_path in link_lst:
            soup = self.establish_connection(link_path)
            sub_soup = soup.find(string = re.compile('^\)')).next_sibling           
            birds = sub_soup.find_all('li')
            top = self.get_top_n(1, birds)
            top_birds.append(top)
        
        path_txt = ''.join([self.path, '.txt'])
        file = open(path_txt, 'a')
        for link in top_birds:
            file.write(''.join([link, '\n']))
        file.close()
         
        return top_birds
        
class retriver():
    
    def __init__(self, base, path_file, path_parquet, time_len, quality_rate, frame_len, hop_len):
        
      self.base = base
      self.path_file = path_file
      self.path_parquet = path_parquet
      self.time_len = time_len
      self.quality_rate = quality_rate
      self.frame_len = frame_len
      self.hop_len = hop_len
      
    def connect(self, url):
        
       session = requests.Session() 
       retry = Retry(total = 10000000, backoff_factor = 2)
       adapter = HTTPAdapter(max_retries = retry)
       session.mount('http://', adapter)
       session.mount('https://', adapter)
        
       r = session.get(url)
       contt = r.content
       
       return contt
    
    def get_table(self, url):
        
        cont = self.connect(url)
        soup = bs(cont, "html5lib")
        
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
    
        # Adjust elevation:
            
        df['Elev. (m)'] = df['Elev. (m)'].apply(lambda x: np.nan if x == '?' else x)
        
        return df

    def cut_longer_audio (self, wave, sample_rate):
        
        rms_wave = librosa.feature.rms(wave, frame_length = self.frame_len, hop_length = self.hop_len)[0]
        peak_pos = find_peaks(rms_wave)[0]
        peak_tr = np.quantile(rms_wave[peak_pos], q = 0.95)
        random_peak = np.random.choice(a = np.where(rms_wave > peak_tr)[0])
        del rms_wave
        
        division_peak = random_peak * self.hop_len
        lower_cut = division_peak - self.time_len/2 * sample_rate
        upper_cut = division_peak + self.time_len/2 * sample_rate
    
        if lower_cut < 0:
            upper_cut += abs(lower_cut)
            lower_cut = 0
        elif upper_cut > len(wave):
            lower_cut -= abs(upper_cut - len(wave))
            upper_cut = len(wave)-1
        
        selected_audio = wave[int(lower_cut) : int(upper_cut)]
        
        
        if selected_audio.shape[0] < self.quality_rate * self.time_len:
            selected_audio = np.concatenate((selected_audio, np.zeros(self.quality_rate*self.time_len-selected_audio.shape[0])))
        
        return selected_audio
    
    @thread_manager
    def get_song(self, url, audio_lst):
        
        data = self.connect(url)
        
        try:
            
            sound = AudioSegment.from_file(BytesIO(data)).set_channels(1).set_frame_rate(self.quality_rate)
            wave = np.array(sound.get_array_of_samples()).astype(float)
            sample_rate = sound.frame_rate
        
            if wave.shape[0] < self.quality_rate * self.time_len:
            
                wave = np.concatenate((wave, np.full(self.quality_rate * self.time_len-wave.shape[0], np.nan)))
            
            else:
                
                wave = self.cut_longer_audio(wave, sample_rate)
            
            xcid = ''.join(['XC', re.search(r'\d+', url)[0]])
            to_df = [xcid] + wave.tolist()
            audio_lst.append(to_df)
       
        except:
            
            xcid = ''.join(['XC', re.search(r'\d+', url)[0]])
            audio_lst.append([xcid] + [np.nan for _ in range(self.time_len * self.quality_rate)])
            
    def clean_rows(self, df):
        
        idx_remove = []
        common = []
        scientific = []
        urls = []
        for row in df.itertuples():
            
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
                url_sound = ''.join([self.base, '/', xc_num, '/download'])
                urls.append(url_sound)
            
            else:
                
                idx_remove.append(row.Index)
                
        return idx_remove, common, scientific, urls
    
    def to_parquet(self, df, name):
        
        path_parquet_bird = ''.join([self.path_parquet, name, '.parquet'])

        '''
        fields = [pa.field('id', pa.string()), pa.field('common_name', pa.string()), 
                  pa.field('scientific_name', pa.string()), pa.field('date', pa.string()),
                  pa.field('time', pa.string()), pa.field('location', pa.string()),
                  pa.field('elevetaion', pa.int32()), pa.field('type', pa.string())]
        fields.extend([pa.field('wf_' + str(n), pa.float32()) for n in range(110250)])
        my_schema = pa.schema(fields)
        '''
        #schema = my_schema, in froma_pandas
        table = pa.Table.from_pandas(df,  preserve_index = False,
                                     nthreads = df.shape[1])
        pq.write_table(table, path_parquet_bird)
    
    def get_data(self, link_path):
        
        # Find last page to parse:
            
        spec_name = re.search(r'(?<=species\/)(.*)', link_path)[0].split('-')
        url = ''.join([self.base, '/explore?query=', spec_name[0], '%20', spec_name[1]])
        cont = self.connect(url)
        soup = bs(cont, "html.parser")
        bar = soup.find('nav', {'class': 'results-pages'})
        last_pg = int(bar.find_all('li')[-2].text)
     
        # Start parsing:
            
        counter = 0
        my_df = pd.DataFrame()
        for n in range(1, last_pg + 1): 
    
            if counter >= 200:
                break
                  
            # Get the table in the website:
            
            main_url = ''.join([self.base, '/explore?query=', spec_name[0], '+', spec_name[1], '&pg=', str(n)])
            try:
                chunk = self.get_table(main_url)
            except:
                warnings.warn('Table lost\nLink: {}'.format(main_url))
                continue
            
            # Work on the rows and keep only the ones needed:
            
            idx_remove, common, scientific, urls = self.clean_rows(chunk)    
            
            # Download audio and get spectrogram:
             
            audio_lst = []
            threads = []
            for url_download in urls:
                t = self.get_song(url_download, audio_lst)
                threads.append(t)
            for process in threads:
                process.join()

            audio_df = pd.DataFrame(audio_lst, columns = ['id'] +['wf_' + str(i) for i in range(self.time_len * self.quality_rate)])
            
            # Adjust DF:
            
            chunk.drop(idx_remove, axis = 0, inplace = True)
            chunk.drop(['Unamed','Common name / Scientific', 'Location', 'Length', 'Recordist', 'Remarks', 'Actions'], 
                       axis = 1, inplace = True)
            chunk['common_name'] = common
            chunk['scientific_name'] = scientific
            
            chunk.columns = ['date', 'time', 'country', 'elevetaion', 'type', 'id', 'common_name', 'scientific_name']
            chunk = chunk[['id', 'common_name', 'scientific_name', 'date', 'time', 'country', 'elevetaion', 'type']]
            
            chunk = chunk.merge(audio_df, how = 'inner', on = 'id')
            
            counter += len(chunk)
            '''
            # Save CSV:
                
            path_csv = ''.join([self.path_file, '.csv'])  
            
            if not os.path.exists(path_csv):     
                chunk.to_csv(path_csv, mode = 'w', header = True, index = False)
            else:
                chunk.to_csv(path_csv, mode = 'a', header = False, index = False)
            '''
            # Add to my_df:
                
            if len(my_df) == 0:
                my_df = chunk
            else:
                my_df = pd.concat([my_df, chunk], axis = 0, ignore_index = True)
        
        # Save table as parquet:

        self.to_parquet(my_df, spec_name[0])
           
    def merge_parquets(self):
        
        parquet_lst = glob(''.join([self.path_parquet, '*.parquet']))
        pq_tables = []
        for f in tqdm(parquet_lst):
            table = pq.read_table(f)
            pq_tables.append(table)
            os.remove(f)
        final_table = pa.concat_tables(pq_tables)
        pq.write_table(final_table, ''.join([self.path_file, '.parquet']), 
                       use_dictionary = True, compression='snappy')
        
       