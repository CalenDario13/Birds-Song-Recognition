import pandas as pd
import numpy as np
import heapq
import os.path
import warnings
from tqdm import tqdm
from collections import defaultdict
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
    """
    This is a wrapper and allows to run in multithread any desired function.
    """
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=function, args=[item for item in args], kwargs=kwargs)
        thread.start()
        return thread
    return wrapper

def process_manager(function, lst, n_core = 5):
    """
    This function allows to run any function in multiprocessing.
    """
    with mp.Pool(n_core) as p:
        list(tqdm(p.imap(function, lst), total = len(lst)))


class top_ten():
    
    """
    This class is developed to dive into the taxonomy structure in order to find the top 10 species
    with the highest number of recordings.
    """
    
    def __init__(self, base, path):
        
        self.base_url = base
        self.path = path
        
    def establish_connection(self, link_path, n = 10000, t = 0.5):
        
        """
        It allows to establish connection with a given web-page and if a failure occurs it tries
        n times every t seconds to get the page. It returns the soup of the web-page.
        
            Output: the soup (got by BeautifulSoup)
        """
        
        session = requests.Session() 
        retry = Retry(total = n, backoff_factor = t)
        adapter = HTTPAdapter(max_retries = retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        
        url = ''.join([self.base_url, link_path])
        r = session.get(url)
        soup = bs(r.content, 'html.parser')
        
        return soup
    
    def get_info(self, soup):
        
        """
        This function is designed to get anuwhere the name of the bird/species/anyelse,
        the number of recordings for it and the link to get it.
        """
        
        name = soup.find('a').text.strip()
        score = int(soup.find('span', {'class': 'recording-count'}).text.strip())
        link_path = soup.find('a')['href']
        
        return name, score, link_path
    
    def get_top_n(self, n, sub_soup):
        
        """
        This function stores the given infromation contained in the sub_soup 
        and returns the top n elements og the given group.
        
            Output: a link as str
        """
        
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
        
    def top_order(self):
        
        """
        This function returns the top 10 orders with more recordings.
        
            Output: a list with ten order links
        """
        
        path_url = '/explore/taxonomy'
        soup = self.establish_connection(path_url)
        taxonomies = soup.find_all('li')
        top_10 = self.get_top_n(10, taxonomies)
        
        return top_10
        
    def top_families(self):
        
        """
        This function returns for each given orderthe family with the most recordings.
        
            Output: a list with ten families links
        """
        
        link_lst = self.top_order()
        
        top_species = []
        for link_path in link_lst:
            soup = self.establish_connection(link_path)
            sub_soup = soup.find('a', {'href': link_path}).next_sibling            
            species = sub_soup.find_all('li')
            top = self.get_top_n(1, species)
            top_species.append(top)
         
        return top_species

    def top_genus(self):
        
        """
        This function returns for eacch given family the genus with the most recordings.
        
            Output: a list with ten genus links
        """
        
        link_lst = self.top_families()
        
        top_sub_species = []
        for link_path in link_lst:
            soup = self.establish_connection(link_path)
            sub_soup = soup.find(string = re.compile('\s\(.*\)')).next_sibling            
            sub_species = sub_soup.find_all('li')
            top = self.get_top_n(1, sub_species)
            top_sub_species.append(top)
         
        return top_sub_species
    
    def top_species(self):
        
        """
        This function returns for eacch given genus the species with the most recordings.
        
            Output: a list with ten species links_path (NOT compelte link).
        """
        
        link_lst = self.top_genus()
        
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
    
    """
    This class is developed to navigate in the website and retrive information finally sotred in a Dataframe.
    """
    
    def __init__(self, base, path_file, path_parquet, time_len, quality_rate, frame_len, hop_len):
        
      self.base = base
      self.path_file = path_file
      self.path_parquet = path_parquet
      self.time_len = time_len
      self.quality_rate = quality_rate
      self.frame_len = frame_len
      self.hop_len = hop_len
      
    def connect(self, url, n = 100000000, t = 2):
        
        """
        It allows to establish connection with a given web-page and if a failure occurs it tries
        n times every t seconds to get the page. It returns the content of the web-page.
            Input:
                url: compelte url to whatevere
            Output: the content of the given web page
        """
        
        session = requests.Session() 
        retry = Retry(total = n, backoff_factor = t)
        adapter = HTTPAdapter(max_retries = retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
         
        r = session.get(url)
        contt = r.content
        
        return contt
    
    def get_table(self, url):
        
        """
        This function get as input a link to a webpage containg dtabase and scrape it.
            Input:
                url: the COMPLETE url to get the table
            Output: a pandas DataFrame with the raw info
        """
        
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
        
        """
        This function reduces the dimension of recordings bigger than a given threshold.
            Input:
                wave. a numpy array which contains the waveform
                sample_rate: the sampel rate as int
                
            Output: a waveform of a given length as numpy array
        """
        
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
        
        """
        This function extract the waveform from recordings.
        It is designed to work in mmultithread (thanks to the wrapper.
            
            Input:
                url: a compelte url from which get the song.
                audio_lst: a list where to save the audioform (it will be tarnsformed in df).
            
            Output: a numpy arry which contains the waveform
        """
        
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
    
    @thread_manager        
    def get_gps_and_back(self, url_gps, gps_back_dic):
        
        """
        This function is developed to get the gps information about each row of the DataFrame 
        and which birds are also in the audio with the main one.
        It is developed to work un multithreading thanks to the wrapper.
            
            Input:
                url_gps: a string containing the COMPLETE url to get gps coordinates.
                gps_back_di: a dictionary where to save the retrived info.
            
            Output: it appends xcid, latiute, longitude and background directly into the dictioanry.
        """
        
        data = self.connect(url_gps)
        soup = bs(data, 'html.parser')
        content_table = soup.find('table', {'class': 'key-value'})
        rows = content_table.find_all('tr')
        
        found = 0
        for el in rows:
           
            try:
                if el.next.text == 'Latitude':
                    gps_back_dic['latitude'].append(el.next.next.next.text)
                    found +=1
                elif el.next.text == 'Longitude':
                    gps_back_dic['longitude'].append(el.next.next.next.text)
                    found+=1
                elif el.next.text == 'Background':
                    backs = el.find('td', {'valign':'top'})
                    try:
                        if backs.text.strip() != 'none':
                            lis = el.find_all('li')
                            birds = ''
                            for li in lis:
                                common_n = li.a.text
                                scientific_n = li.find('span', {'class':'sci-name'}).text
                                
                                b = ' --- '.join([common_n.strip(), scientific_n.strip()])
                                if birds:
                                    birds = birds + '; ' + b
                                else:
                                    birds = b
                            gps_back_dic['background'].append(birds)
                        else:
                            gps_back_dic['background'].append(np.nan)
                    except:
                        gps_back_dic['background'].append(np.nan)       
            except:
                continue
            
        
        idn = re.search(r'(?<=\/)\d+', url_gps)[0]
        xc_id = ''.join(['XC', idn])
        gps_back_dic['id'].append(xc_id)
    
    def clean_rows(self, df):
        
        """
        This function helps to drop rows that contains unuseful or difficult info to be used.
            Input:
                df: the chunk of df downloaded form the current page to be clean
            Output:
                idx_remove: a list od indexes to drop in the DataFrame
                common: a list with the birds' common names
                scientific: a list with the birds' scientific names
        """
        
        idx_remove = []
        common = []
        scientific = []
        urls = []
        for row in df.itertuples():
            
            try:
                seen = re.search(r'(?<=bird-seen:)\w+', row.Remarks)[0]
            except:
                seen = 'no'
                
            #score = row.Actions
            
            if seen == 'yes' and (row._2 != '(?) Identity unknown' or row._2 != 'Soundscape') and row.Length < 120:
                
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
                url_sound = ''.join([self.base, '/', xc_num])
                urls.append(url_sound)
            
            else:
                
                idx_remove.append(row.Index)
                
        return idx_remove, common, scientific, urls
    
    def to_parquet(self, df, name):
        
        """
        This function save the final DataFrame in parquet format into a fiven destination path
            
            Input:
                df: the final dataframe with all the rows for a given species
                name: a stiring with the name of the species
        """
        
        path_parquet_bird = ''.join([self.path_parquet, name, '.parquet'])

        table = pa.Table.from_pandas(df,  preserve_index = False, nthreads = df.shape[1])
        pq.write_table(table, path_parquet_bird)
    
    def get_data(self, link_path):
        
        """
        This function calls the previous one and integrets them in order to create the DataFrame.
        After getting the DataFrame, it modifies it in order to make it readable and clean.
        
            Input:
                link_path: a stirng that represents ONLY the path to a given species (NOT the base url)
            Output: it concatenate a partial DataFrame to the final one.
        """
        
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
            gps_back_dic = defaultdict(list)
            threads_audio = []
            threads_gps = []
            for url_download in urls:
                url_download_audio = ''.join([url_download, '/download'])
                t_a = self.get_song(url_download_audio, audio_lst)
                t_g = self.get_gps_and_back(url_download, gps_back_dic)
                threads_audio.append(t_a)
                threads_gps.append(t_g)
            for process_a, process_g in zip(threads_audio, threads_gps):
                process_a.join()
                process_g.join()

            audio_df = pd.DataFrame(audio_lst, columns = ['id'] +['wf_' + str(i) for i in range(self.time_len * self.quality_rate)])
            if len(gps_back_dic) == 0:
                gps_df = pd.DataFrame(columns = ['id', 'latitude', 'longitude', 'background'])
            else:
                gps_df = pd.DataFrame.from_dict(gps_back_dic)       
            
            # Adjust DF:
            
            chunk.drop(idx_remove, axis = 0, inplace = True)
            chunk.drop(['Unamed','Common name / Scientific', 'Location', 'Length', 'Recordist', 'Remarks', 'Actions'], 
                       axis = 1, inplace = True)
            chunk['common_name'] = common
            chunk['scientific_name'] = scientific
            
            chunk.columns = ['date', 'time', 'country', 'elevetaion', 'type', 'id', 'common_name', 'scientific_name']
            chunk = chunk.merge(gps_df, how = 'inner', on = 'id')
            chunk = chunk[['id', 'common_name', 'scientific_name', 'date', 'time', 'country', 
                           'latitude', 'longitude', 'elevetaion', 'background', 'type']]
            
            chunk = chunk.merge(audio_df, how = 'inner', on = 'id')
            
            counter += len(chunk)
            """
            # Save CSV:
                
            path_csv = ''.join([self.path_file, '.csv'])  
            
            if not os.path.exists(path_csv):     
                chunk.to_csv(path_csv, mode = 'w', header = True, index = False)
            else:
                chunk.to_csv(path_csv, mode = 'a', header = False, index = False)
            """
            # Add to my_df:
                
            if len(my_df) == 0:
                my_df = chunk
            else:
                my_df = pd.concat([my_df, chunk], axis = 0, ignore_index = True)
        
        # Save table as parquet:

        self.to_parquet(my_df, spec_name[0])
           
    def merge_parquets(self):
        
        """
        This function merge all the parquets files (one for each species) in a bigger one.
        """
        
        parquet_lst = glob(''.join([self.path_parquet, '*.parquet']))
        pq_tables = []
        for f in tqdm(parquet_lst):
            table = pq.read_table(f)
            pq_tables.append(table)
            os.remove(f)
        final_table = pa.concat_tables(pq_tables)
        pq.write_table(final_table, ''.join([self.path_file, '.parquet']), 
                       use_dictionary = True, compression='snappy')





