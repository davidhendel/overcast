import os
import sys
import time
import requests
_DIR = './'

'''
Purpose: 
    Script to query New Mexico Skies for all-sky images 
    and cloud sensor data every five minutes.

Usage: python get_images.py [DATEDIR]

Returns: nothing
'''

#Local directory base
_DATADIR = _DIR 
#Image folder at NMSkies
_URLBASE = 'https://www.nmskies.com/images/'
#Rename images with dictionary for convenience
namedict={
    'lowres':'AllSkyImage.jpg',
    'hires':'allsky.jpg',
    'csens':'AAG_ImageCloudCondition.png'}

def get_image(imname):
    '''
    Get an image and write one with a more informative name and 
    a timestamp in the DATADIR folder
    '''
    img_data = requests.get(_URLBASE+namedict[imname]).content
    outfile_name = _DATADIR+imname+'_'+time.strftime('%Hh%Mm%Ss_%Y-%m-%d_%Z')+'.'+namedict[imname].split('.')[-1]
    with open(outfile_name, 'wb') as handler:
        handler.write(img_data)

if __name__ == '__main__':

    if len(sys.argv)>0:
        try: 
            os.mkdir(str(sys.argv[1]))
            _DATADIR = _DATADIR+str(sys.argv[1])+'/'
        except: 
            print('Error: directory already exists')
            sys.exit()

    while True:
        for im in namedict.keys():
            get_image(im)
        time.sleep(60)
