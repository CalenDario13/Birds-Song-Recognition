# Birds-Song-Recognition

![](img/bird_img.jpg)

This project has started to classify the 10 most common species of birds from their call. The aim is to develope an application which is able to recognise the most common birds songs.
The files are retrived from the https://www.xeno-canto.org/, an opensource website where people can load their recordings.
### HOW TO RUN THE CODE 
1. **new_retriver.py** is used to get the most common birds, thanks to the top_ten class and download the files thanks to the class retriver.
2. **preprocessing.py** is used to process data and clean then as describe on _Birds-Song-Recognition-report_
3. **model.py** prepares the data and evaluate the model
4. **main_birds.py** execute the program. 
5. **Header.py** contains the paths, download it and change the paths inside with you own ones. This file won't be updated.
6. **Birds_Map.html** is a html interactive map

Link to Download our DataSet:

- CSV File: https://drive.google.com/file/d/1rrKAuEhpj9qNmZ6IcgIG2_r0HanKim2Q/view
- Parquet File: https://drive.google.com/file/d/183haB0l8z-wro1DYom9ZpyRsfDPJ-W3a/view

### Don't touch or use data_retriver_birds.py
