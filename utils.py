import math
from datetime import datetime
import pandas as pd
from geopy.exc import GeocoderTimedOut 
from geopy.geocoders import Nominatim 
import numpy as np
from collections import defaultdict



# function to find the coordinate 
# of a given city  
def findGeocode(city): 
       
    # try and catch is used to overcome 
    # the exception thrown by geolocator 
    # using geocodertimedout   
    try: 
          
        # Specify the user_agent as your 
        # app name it should not be none 
        geolocator = Nominatim(user_agent="your_app_name") 
          
        return geolocator.geocode(city) 
      
    except GeocoderTimedOut: 
          
        return findGeocode(city)     
  
# each value from city column 
# will be fetched and sent to 
# function find_geocode 

def lat_lon(dataf):
# declare an empty list to store 
# latitude and longitude of values  
# of city column 
    longitude = [] 
    latitude = [] 
    d = defaultdict(list)
    for coun in (dataf["country"]): 
        if coun in list(d.keys()):
            latitude.append(d[coun][0])
            longitude.append(d[coun][1])
        else:
            loc = findGeocode(coun) 

            
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
    dataf["longitude"] = longitude 
    dataf["latitude"] = latitude
    return dataf


def calc_gio_not(df1):
    gio_not = []
    sun = Sun()
    for oss in df1.itertuples():
        date = oss[1]
        time = oss[2]
        lon = oss[4]
        lat = oss[5]
        coords = {'longitude' : lon, 'latitude' : lat }
        d_alb = sun.calcSunTime(date,coords,True)
        ore_alb,min_alb = d_alb['hr'],d_alb['min']

        d_tram = sun.calcSunTime(date,coords,False)
        ore_tram,min_tram = d_tram['hr'],d_tram['min']
        if time == '?':
            gio_not.append('Nan')
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
    df1["gio_not"] = gio_not
    return df1



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

def emisphere(lat):
    if lat >= 0:
        e = 'north'
    else:
        e = 'sud'
    return e


def season(data, HEMISPHERE):
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


def calc_season(df1):
    season_lis = []
    for oss in df1.itertuples():
        date = oss[1]
        lat = oss[5]
        e = emisphere(lat)
        s = season(date,e)
        season_lis.append(s)
    df1["season"] = season_lis
    return df1