import pandas as pd
import numpy as np
from collections import defaultdict
import math
from datetime import datetime
from geopy.exc import GeocoderTimedOut 
from geopy.geocoders import Nominatim
from tqdm import tqdm


class Sun:

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
        self.df = pd.concat([self.df.iloc[:,0:7], type_df, self.df.iloc[:,8:]], axis = 1)

    
    # function to find the coordinate 
    # of a given city  
    def findGeocode(self,city): 
        
        # try and catch is used to overcome 
        # the exception thrown by geolocator 
        # using geocodertimedout   
        try: 
            
            # Specify the user_agent as your 
            # app name it should not be none 
            geolocator = Nominatim(user_agent="your_app_name") 
            
            return geolocator.geocode(city) 
        
        except GeocoderTimedOut: 
            
            return self.findGeocode(city) 

    def lat_lon(self):
    # declare an empty list to store 
    # latitude and longitude of values  
    # of city column 
        longitude = [] 
        latitude = [] 
        d = defaultdict(list)
        for coun in tqdm(self.df["country"], desc = 'Calculate lat-long'): 
            if coun in list(d.keys()):
                latitude.append(d[coun][0])
                longitude.append(d[coun][1])
            else:
                loc = self.findGeocode(coun) 

                
                if loc:   
                    # coordinates returned from  
                    # function is stored into 
                    # two separate list 
                    latitude.append(loc.latitude) 
                    longitude.append(loc.longitude) 
                    d[coun].append(loc.latitude)
                    d[coun].append(loc.longitude)
                
                # if coordinate for a city not 
                # found, insert "NaN" indicating  
                # missing value  
                else: 
                    latitude.append(np.nan) 
                    longitude.append(np.nan)
                    d[coun].append(loc.latitude)
                    d[coun].append(loc.longitude) 
        #Showing the output produced as dataframe.


        # now add this column to dataframe 
        self.df.insert(6,"longitude",longitude) 
        self.df.insert(7,"latitude",latitude)


    def calc_gio_not(self):
        gio_not = []
        sun = Sun()
        for oss in tqdm(self.df.iloc[:,0:13].itertuples(), desc = 'Calculate day-night', total=len(self.df)):
            date = oss.date
            time = oss.time
            lon = oss.longitude
            lat = oss.latitude
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
        self.df.insert(8,"gio_not",gio_not)
    

    def emisphere(self,lat):
        if lat >= 0:
            e = 'north'
        else:
            e = 'sud'
        return e

    def season(self,data, HEMISPHERE):
        s = Sun()
        date = s.getCurrentUTC(data)
        md = date[1] * 100 + date[0]

        if ((md > 320) and (md < 621)):
            s = 0 #spring
        elif ((md > 620) and (md < 923)):
            s = 1 #summer
        elif ((md > 922) and (md < 1223)):
            s = 2 #fall
        else:
            s = 3 #winter

        if not HEMISPHERE == 'north':
            s = (s + 2) % 3
        return s


    def calc_season(self):
        season_lis = []
        for oss in tqdm(self.df.iloc[:,0:14].itertuples(), desc = 'Calculate Season', total=len(self.df)):
            date = oss.date
            lat = oss.latitude
            e = self.emisphere(lat)
            s = self.season(date,e)
            season_lis.append(s)
        self.df.insert(9,"season",season_lis)

    def generate_final_db(self):
        self.add_type_columns()
        self.lat_lon()
        self.calc_gio_not()
        self.calc_season()
        return self.df
    

        
        
        
        


