import pandas as pd
import numpy as np
from collections import defaultdict
import math
from datetime import datetime
from tqdm import tqdm
import librosa

from sklearn.preprocessing import scale

class Sun:
    """This class is made up of several functions, 
    the main ones take as input latitude,longitude and a date and calculate the sunrise and sunset times, 
    while the third takes as input a date which is a string and returns a datatime."""

    def getSunriseTime( self, data, coords ):
        return self.calcSunTime( data,coords, True )

    def getSunsetTime( self, data, coords ):
        return self.calcSunTime( data,coords, False )

    def getCurrentUTC( self, data):
        data = data.split('-')
        if data[2] == '00':
            data[2] = '01'
        data = '-'.join(data)
        data = datetime.strptime(data, '%Y-%m-%d')
        return [ data.day, data.month, data.year ]

    def calcSunTime( self, data, coords, isRiseTime, zenith = 90.8 ):

        # isRiseTime == False, returns sunsetTime

        day, month, year = self.getCurrentUTC(data)

        longitude = coords['longitude']
        latitude = coords['latitude']

        TO_RAD = math.pi/180

        #1. first calculate the day of the year
        N1 = math.floor(275 * month / 9)
        N2 = math.floor((month + 9) / 12)
        N3 = (1 + math.floor((year - 4 * math.floor(year / 4) + 2) / 3))
        N = N1 - (N2 * N3) + day - 30

        #2. convert the longitude to hour value and calculate an approximate time
        lngHour = longitude / 15

        if isRiseTime:
            t = N + ((6 - lngHour) / 24)
        else: #sunset
            t = N + ((18 - lngHour) / 24)

        #3. calculate the Sun's mean anomaly
        M = (0.9856 * t) - 3.289

        #4. calculate the Sun's true longitude
        L = M + (1.916 * math.sin(TO_RAD*M)) + (0.020 * math.sin(TO_RAD * 2 * M)) + 282.634
        L = self.forceRange( L, 360 ) #NOTE: L adjusted into the range [0,360)

        #5a. calculate the Sun's right ascension

        RA = (1/TO_RAD) * math.atan(0.91764 * math.tan(TO_RAD*L))
        RA = self.forceRange( RA, 360 ) #NOTE: RA adjusted into the range [0,360)

        #5b. right ascension value needs to be in the same quadrant as L
        Lquadrant  = (math.floor( L/90)) * 90
        RAquadrant = (math.floor(RA/90)) * 90
        RA = RA + (Lquadrant - RAquadrant)

        #5c. right ascension value needs to be converted into hours
        RA = RA / 15

        #6. calculate the Sun's declination
        sinDec = 0.39782 * math.sin(TO_RAD*L)
        cosDec = math.cos(math.asin(sinDec))

        #7a. calculate the Sun's local hour angle
        cosH = (math.cos(TO_RAD*zenith) - (sinDec * math.sin(TO_RAD*latitude))) / (cosDec * math.cos(TO_RAD*latitude))

        if cosH > 1:
            return {'status': False, 'msg': 'the sun never rises on this location (on the specified date)'}

        if cosH < -1:
            return {'status': False, 'msg': 'the sun never sets on this location (on the specified date)'}

        #7b. finish calculating H and convert into hours

        if isRiseTime:
            H = 360 - (1/TO_RAD) * math.acos(cosH)
        else: #setting
            H = (1/TO_RAD) * math.acos(cosH)

        H = H / 15

        #8. calculate local mean time of rising/setting
        T = H + RA - (0.06571 * t) - 6.622

        #9. adjust back to UTC
        UT = T - lngHour
        UT = self.forceRange( UT, 24) # UTC time in decimal format (e.g. 23.23)

        #10. Return
        hr = self.forceRange(int(UT), 24)
        min = round((UT - int(UT))*60,0)

        return {
            'status': True,
            'decimal': UT,
            'hr': hr,
            'min': min 
        }

    def forceRange( self, v, max ):
        # force v to be >= 0 and < max
        if v < 0:
            return v + max
        elif v >= max:
            return v - max

        return v

class Cleaner():
    
    """
    This class adjustes some issues that came up after reading the DataFrame for the first time.
        Input:
            df: the pandas DataFrame containg all the info.
    """
    
    def __init__(self, df):
        
        self.df = df
        
    def transform_columns(self):
        
        """
        This function adjusts some troubles due to wrong data types 
        and add an extra column to check if a bird is alone in a recording
        """
        
        self.df.elevetaion = self.df.elevetaion.apply(lambda x: int(x) if (x and x.isdigit()) else np.nan)
        self.df.latitude = self.df.latitude.apply(lambda x: np.nan if x == 'Not specified' else float(x) * np.pi/180)
        self.df.longitude = self.df.longitude.apply(lambda x: np.nan if x == 'Not specified' else float(x) * np.pi/180)
        is_alone = self.df.background.apply(lambda x: 'yes' if not x else 'no') # alone = 0
        self.df.insert(9, 'is_alone', is_alone)
        
        self.df.loc[self.df['common_name'] == '(?) Mallard', 'common_name'] = 'Mallard'

    def clean_type(self, row):
        
        """
        This function check the tags present in agiven column and assign each to the corresponding list.
            Input: a row of the pandas df.
        """
        
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
        
        for tag in row.split(','):
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
        
        """
        Given the length of the list the function decides what to do with it, 
        ongly if the list is made of only one arguments it is transformed in string.
        Otherwise, it returns nan.
            Input: 
                lst: a list of strings (tags) of given ctegory (see the clean_type function for the classes)
            output:
                new: a string containing the tag
        """
        
        if len(lst) > 1:
            new = np.NaN
            return new
        elif len(lst) == 1:
            new = lst[0]
            return new
        elif len(lst) == 0:
            new = np.NaN
            return new
    
    def add_type_columns(self):
        
        """
        This function analyze tags in type column, for each row 
        and generates a column for each tag with the given tag if exists or nan
        """
        
        tags = defaultdict(list)
        for row in tqdm(self.df['type'], desc = 'Generate type columns'):
            
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
        self.df = pd.concat([self.df.iloc[:,0:10], type_df, self.df.iloc[:,11:]], axis = 1)


    def calc_gio_not(self):
        """This function takes as input the date, time, latitude and longitude of the observation of a bird,
        recalls the functions of the sun class to calculate the time of sunrise and sunset and checks if the observation time is between the two.
        the function returns a string 'day' if it is true otherwise it returns the string 'night'."""
        gio_not = []
        sun = Sun()
        for oss in tqdm(self.df.iloc[:,0:15].itertuples(), desc = 'Calculate day-night', total=len(self.df)):
            
            date = oss.date
            time = oss.time
            lon = oss.longitude
            lat = oss.latitude
            
            try:
     
                coords = {'longitude' : lon, 'latitude' : lat }
                d_alb = sun.calcSunTime(date,coords,True)
                ore_alb,min_alb = d_alb['hr'],d_alb['min']
    
                d_tram = sun.calcSunTime(date,coords,False)
                ore_tram,min_tram = d_tram['hr'],d_tram['min']
                if time == '?':
                    gio_not.append(np.NaN)
                else:
                    time_ore,time_min = time.split(":")
    
                    if time_min == '00am':
                        time_min = '00'
                    elif time_min == '30am':
                        time_min = '30'
                    elif time_min == '30pm':
                        time_min = '30'
    
                    time_ore = int(time_ore)
                    time_min = int(time_min)
    
                    if time_ore < ore_tram and time_ore > ore_alb:
                        gio_not.append('giorno') 
                    elif time_ore == ore_alb:
                        if time_min >= min_alb:
                            gio_not.append('giorno')
                        else:
                            gio_not.append('notte') 
                    elif time_ore == ore_tram:
                        if time_min < min_tram:
                            gio_not.append('giorno')
                        else:
                            gio_not.append('notte')
                    else:
                        gio_not.append('notte')
            except:
          
                gio_not.append(np.NaN)

        self.df.insert(9,"gio_not",gio_not)
    

    def emisphere(self,lat):
        """This function takes as input the latitude of the place where the bird was observed 
        if it is greater than 0 return that was observed in the northern hemisphere otherwise in the southern one."""
        if lat >= 0:
            e = 'north'
        else:
            e = 'sud'
        return e

    def season(self,data, HEMISPHERE):
        """This function takes date and hemisphere as input.
        From the date it extracts the day and month in order to calculate 
        in which season of the year the observation took place."""
        
        seasons = {0: 'spring',
                   1: 'summer',
                   2: 'fall',
                   3: 'winter'}
        s = Sun()
        date = s.getCurrentUTC(data)
        md = date[1] * 100 + date[0]

        if ((md > 320) and (md < 621)):
            se = 0 #spring
        elif ((md > 620) and (md < 923)):
            se = 1 #summer
        elif ((md > 922) and (md < 1223)):
            se = 2 #fall
        else:
            se = 3 #winter

        if not HEMISPHERE == 'north':
            se = (se + 2) % 3
    
        return seasons[se]

    def calc_season(self):
        "Final function that uses the previous two to save season information in the DataFrame"
        season_lis = []
        for oss in tqdm(self.df.iloc[:,0:16].itertuples(), desc = 'Calculate Season', total=len(self.df)):
            date = oss.date
            lat = oss.latitude
            if not lat:
                season_lis.append(np.NaN)
            else:
                e = self.emisphere(lat)
                s = self.season(date,e)
                season_lis.append(s)
        self.df.insert(10,"season",season_lis)

    def binning_elev(self, el):
        
        """
        This function helps to transform elevation in a dummy variable, 
        for each passed row it determinates in which bin the value is.
            Input:
                el: an int which represents the elevation
            Output:
                binn: a string representing the given bin
        """
        
        if el <= 240:
            binn = 'bassa'
        elif 240 < el <= 500:
            binn = 'media'
        else:
            binn = 'alta'
            
        return binn
   
    def generate_final_db(self):
        
        """
        This function calls all the previous ones and adjusts the DataFrame.
            output:
                df: the cleaned df
        """
 
        self.add_type_columns()
        self.transform_columns()
        self.df['elevetaion'] = self.df.elevetaion.apply(lambda x: self.binning_elev(x))
        self.df = self.df.rename({'elevetaion':'elevation'}, axis=1)
        self.calc_gio_not()
        self.calc_season()
       
        return self.df
    
class Audio_Processing():
    
    """
    This class works only on the waveform and transforms it thanks to Fourier transform 
    or MCCS, depending on what we want.
    """
    
    def __init__(self, df, quality_rate, hop_length, bins, low_cut, high_cut):
        
        self.df = df.iloc[:,17:]
        is_nan = self.df.isnull()
        nan_rows = is_nan.all(axis = 1)
        self.df = self.df[[not i for i in nan_rows]]
        self.df = self.df.fillna(0)
        self.other_df = df.iloc[:,:17]

        self.tempo = [i/22050 for i in range(df.shape[1])]
        self.dt = 1/quality_rate
        self.quality_rate = quality_rate
        self.hop_length = hop_length
        self.bins = bins
        self.low_cut = low_cut
        self.high_cut = high_cut
        
    def fft_filter(self, signal):

        """ 
        The function use the fast fourier transform to obtain the frequencies power spectrum for each audio, 
        then it define a threshold given by average + standard deviation of the power spectrum to filter 
        some noise from the audio.
        Input:
                signal: the audio wave form 
        Output:
                f_filt: the filtered signal
        """

        n = len(self.tempo) 
        
        # Calculate Power Spectrum:  
        fhat = np.fft.fft(signal, n)
        PSD = fhat * np.conj(fhat) / n 
        L = np.arange(1, np.floor(n/2), dtype = int) 
        
        
        # Get info to use to filter:
        freq = (1/(self.dt * n)) * np.arange(n)
        av, sd = PSD[L].mean(), PSD[L].std() # calcola avg e sd dei PSD
        
        # Filter the signal:
        indices = (PSD > av + sd) & (freq >= 4500) # Lista di filtri
        fhat_to_inverse = fhat * indices # Filtraggio fourier
        f_filt = np.fft.ifft(fhat_to_inverse) # Trasformata inversa (da frequenze a tempo)
    
        return f_filt 
      
    def get_mel(self, rows):

        """
        The function generates a set of MEL cepstral coefficients and the 
        Delta-spectral cepstral coefficients
        Input:
                rows: the audio signal once preprocessed
        Output:
                mfcc_concat: a numpy array where are concatenated mfcc and delta-scc
        """
        
        # Generate MEL and Delta:
        signal = np.array(rows, dtype = np.float32) 
        signal = np.absolute(self.fft_filter(signal)) 
        mfcc = librosa.feature.mfcc(y = signal, sr = self.quality_rate, n_fft = 2048,
                                    hop_length = self.hop_length, n_mfcc = 20) 
        mfcc_delta = librosa.feature.delta(mfcc) 
        
        # Flatten and concatenate:
        mfcc = mfcc.flatten('C') 
        mfcc_delta = mfcc_delta.flatten('C')
        mfcc_concat = np.concatenate([mfcc, mfcc_delta]) 

        return mfcc_concat
        
    def return_ffts(self, ffts_len, decibel = False, mel_scale = True):
    
        """Returns the fft in original scale or decibel. The len of the fft should be specified. 
           Can either be decibel of Mel scale. Acts on the waveform part of df.
           Input:
                   ffts_len: integer len of the final array
                   decibel: bool value , if returning in decibel the spectrum
                   mel_scale: bool value , if returning in Mel scale the spectrum
            Output:
                    ffts: matrix of the transformed waveforms
        """
    
        ffts = np.empty((self.df.shape[0], ffts_len))
        wf_matrix = self.df.values
        for i in tqdm(range(wf_matrix.shape[0]), desc = 'Fourier Transformation'):
            last_idx = np.sum(~np.isnan(wf_matrix[i]))
            ffts[i] = np.abs(np.fft.fft(wf_matrix[i, :last_idx], n=ffts_len))
        ffts = ffts[:, : ffts.shape[1]//2]
        ffts = ffts[:, self.low_cut:self.high_cut]
        if decibel:
            ffts = np.log(1+ffts)
        elif mel_scale:
            ffts = 2595*np.log(1+ffts/700)
            
        return ffts

    def eval_spectral_centroid(self, freq):
        """ Function that evaluates the spectral centroid.
        Input:
                freq: array of a single spectrum
        Output:
                centroid: flaot with the centroid of the spectrum
        """
        
        return np.sum(np.arange(self.low_cut, self.high_cut, 1)*freq)/np.sum(freq)    

    def bin_data(self, original_idx = False):
    
        """Function that returns the binned data in matrix form. Original_idx returns also the indices in the original scale.
        Acts on the waveform part of df in matrix form.
        Input:
              original_idx: bool, if returning or not the original index scale
        Output:
                binned_data: matrix of the binned data
                original_idx: array with the original scale if in input original_idx=True

        """
        
        bins_idx = np.linspace(0, self.df.shape[1], self.bins+1).astype(int)
        binned_data = np.empty((self.df.shape[0], self.bins))
    
        for i in tqdm(range(1, self.bins + 1), desc = 'Get Bins'):
            binned_data[:,i-1] = np.mean(self.df[:,bins_idx[i-1]:bins_idx[i]], axis=1)
            #binned_data[:,i-1] = np.quantile(self.df[:,bins_idx[i-1]:bins_idx[i]],0.99, axis=1)
            #binned_data[:,i-1] = np.max(self.df[:,bins_idx[i-1]:bins_idx[i]], axis=1)
            
        if original_idx:
            return binned_data, bins_idx[:-1]
        else:
            return binned_data
    
    def transform_df(self, mel = False):
        
        """
        This function get the orginal waveform and trasform it to mels if mel = True or 
        if mel = False it does the fourier tansformation, bin the data and scale them, then adds
        also the centroids.
        
            Output:
                final: a dataframe with the the tansformed columns (concerning only the waveform).
        """
        
        if mel:
            tqdm.pandas()
            df = self.df.progress_apply(lambda row: self.get_mel(row), axis = 1, result_type = 'expand')
            df.columns = ['mel_' + str(i) for i in range(df.shape[1])]
            
            final = pd.concat([self.other_df, df], axis = 1)
            
        else:
            ffts_len = self.df.shape[1]
            self.df= self.return_ffts(ffts_len) 

            final = self.bin_data()
            final = scale(final, axis = 1)
            final = pd.DataFrame(final, columns = ['bin_' + str(i) for i in range(final.shape[1])])
            final['centroids'] = np.apply_along_axis(self.eval_spectral_centroid, 1, self.df)
            final = pd.concat([self.other_df,  final], axis = 1)
         
        return final
        


