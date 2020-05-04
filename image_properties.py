import numpy as np
import matplotlib.pyplot as plt
import shutil
from numpy.lib.stride_tricks import as_strided

def subimage(im, shape=(80,80))
    '''Reshape image into a 3d array of subimages of size given by shape.
    Currently does not check if the image can evenly divided that way!'''
    xx = as_strided(im, shape=(40, 40, 80, 80), strides=im.strides+im.strides)
    
def entropy(subim):
    '''Comput the Shannon entropy of an image'''
    p = np.histogram(subim,bins=np.linspace(0,255,256), density=True)[0]
    return -sum(p[p>0]*np.log(p[p>0]))

#some cloudy hires - moon is up, will need to cut it out
cloudy_filenames=[
'2020-04-06/hires_00h01m58s_2020-04-07_EDT.jpg',
'2020-04-06/hires_00h06m08s_2020-04-07_EDT.jpg',
'2020-04-06/hires_00h11m20s_2020-04-07_EDT.jpg',
'2020-04-06/hires_00h18m35s_2020-04-07_EDT.jpg',
'2020-04-06/hires_00h24m49s_2020-04-07_EDT.jpg',
'2020-04-06/hires_04h06m08s_2020-04-07_EDT.jpg',
'2020-04-06/hires_04h12m22s_2020-04-07_EDT.jpg',
'2020-04-06/hires_04h17m34s_2020-04-07_EDT.jpg',
'2020-04-06/hires_04h22m47s_2020-04-07_EDT.jpg',
'2020-04-06/hires_04h28m00s_2020-04-07_EDT.jpg',
]

#some clear hires - no moon in these
clear_filenames=[
'2020-04-03/hires_06h54m10s_2020-04-04_EDT.jpg',
'2020-04-03/hires_07h00m33s_2020-04-04_EDT.jpg',
'2020-04-03/hires_07h04m48s_2020-04-04_EDT.jpg',
'2020-04-03/hires_07h10m09s_2020-04-04_EDT.jpg',
'2020-04-03/hires_07h17m36s_2020-04-04_EDT.jpg',
]

for f in clear_filenames:
    shutil.copy(f, './test_images')
for f in cloudy_filenames:
    shutil.copy(f, './test_images')

#load some images, cut to central 1600x1600 area
im_clear = plt.imread('2020-04-02/hires_05h44m35s_2020-04-03_EDT.jpg') [536:2136,1204:2804,0]
im_cloudy = plt.imread('2020-04-06/hires_00h01m58s_2020-04-07_EDT.jpg')[536:2136,1204:2804,0]

#check out images
plt.figure()
plt.subplot(121)
plt.imshow(im_clear, vmin=0, vmax=127, cmap='gray')
plt.title('clear')
plt.subplot(122)
plt.imshow(im_cloudy, vmin=0, vmax=127, cmap='gray')
plt.text(140,1300,'Moon')
plt.title('cloudy')


#Plot image grid
for i in np.arange(4):
    plt.subplot(2,2,i+1)
    plt.imshow(plt.imread(clear_filenames[i])[536:2136,1204:2804,0])
    plt.xticks([])
    plt.yticks([])
plt.suptitle('Clear')
for i in np.arange(9):
    plt.subplot(3,3,i+1)
    plt.imshow(plt.imread(cloudy_filenames[i])[536:2136,1204:2804,0])
    plt.xticks([])
    plt.yticks([])
plt.suptitle('Cloudy')

# # # # # # # # # # # # # # # # # # # # # # # #
#plot histograms of pixel values
plt.subplot(121)
for i in np.arange(4):
    clear_artist = plt.hist(plt.imread(clear_filenames[i])[536:1336,1204:2804,0].flatten(), 
                            color='k', bins=50, log=True, histtype='step', lw=3, alpha=.5)
for i in np.arange(9):
    cloudy_artist = plt.hist(plt.imread(cloudy_filenames[i])[536:1336,1204:2804,0].flatten(), 
                            color='r', bins=50, log=True, histtype='step', lw=3, alpha=.5)
plt.ylabel('Pixel counts')
plt.xlabel('Pixel values')
plt.legend((clear_artist[-1][0], cloudy_artist[-1][0]), ('Clear', 'Cloudy'))

#plot some global statistics
plt.subplot(122)
for i in np.arange(5):
    im_clear  = plt.imread(clear_filenames[i])[536:1336,1204:2804,0]
    clear_vals  = im_clear.flatten()
    clear_artist = plt.scatter(np.median(clear_vals), np.std(clear_vals), c='k', s=20)
        
for i in np.arange(10):
    im_cloudy = plt.imread(cloudy_filenames[i])[536:1336,1204:2804,0]
    cloudy_vals  = im_cloudy.flatten()
    cloudy_artist = plt.scatter(np.median(cloudy_vals), np.std(cloudy_vals), c='r', s=20)
plt.legend((clear_artist, cloudy_artist), ('Clear', 'Cloudy'))
plt.xlabel('Median pixel value')
plt.ylabel('Standard deviation of pixel values')
# # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # # # #
#plot some local statistics - divide into 80x80 patches
for i in np.arange(5):
    #clear
    im  = plt.imread(clear_filenames[i])[536:1336,1204:2804,0]
    #trick to get subimages
    subim  = as_strided(im, shape=(800, 1600, 80, 80), strides=im.strides+im.strides)[::80,::80].reshape(200,80,80)
    entropyarr = np.zeros(len(subim))
    for j in np.arange(len(subim)):
        entropyarr[j] = entropy(subim[j])

    plt.subplot(221)
    clear_artist = plt.scatter(np.median(subim, axis=(1,2)), entropyarr, c='k', alpha=0.4, s=10, edgecolor='none')
    plt.ylabel('Entropy')
    plt.subplot(223)
    plt.scatter(np.median(subim, axis=(1,2)), np.std(subim, axis=(1,2)), c='k', alpha=0.4, s=10, edgecolor='none')
    plt.ylabel('Standard deviation')
    plt.xlabel('Median')
    plt.subplot(224)
    plt.scatter(entropyarr, np.std(subim, axis=(1,2)), c='k', alpha=0.4, s=10, edgecolor='none')
    plt.xlabel('Entropy')

for i in np.arange(10):
    #clear
    im  = plt.imread(cloudy_filenames[i])[536:1336,1204:2804,0]
    #trick to get subimages
    subim  = as_strided(im, shape=(800, 1600, 80, 80), strides=im.strides+im.strides)[::80,::80].reshape(200,80,80)
    entropyarr = np.zeros(len(subim))
    for j in np.arange(len(subim)):
        entropyarr[j] = entropy(subim[j])
        
    plt.subplot(221)
    cloudy_artist = plt.scatter(np.median(subim, axis=(1,2)), entropyarr, c='r', alpha=0.4, s=10, edgecolor='none')
    plt.ylabel('Entropy')
    plt.subplot(223)
    plt.scatter(np.median(subim, axis=(1,2)), np.std(subim, axis=(1,2)), c='r', alpha=0.4, s=10, edgecolor='none')
    plt.ylabel('Standard deviation')
    plt.xlabel('Median')
    plt.subplot(224)
    plt.scatter(entropyarr, np.std(subim, axis=(1,2)), c='r', alpha=0.4, s=10, edgecolor='none')
    plt.xlabel('Entropy')
plt.legend((clear_artist, cloudy_artist), ('Clear', 'Cloudy'))


# # # # # # # # # # # # # # # # # # # # # # # #




if 0:
    plt.figure()
    plt.subplot(121)
    plt.imshow(im_clear)
    plt.plot([(s[1]//2-800),(s[1]//2+800)],[(s[0]//2-800),(s[0]//2-800)],c='r')
    plt.plot([(s[1]//2-800),(s[1]//2+800)],[(s[0]//2+800),(s[0]//2+800)],c='r')
    plt.plot([(s[1]//2-800),(s[1]//2-800)],[(s[0]//2-800),(s[0]//2+800)],c='r')
    plt.plot([(s[1]//2+800),(s[1]//2+800)],[(s[0]//2-800),(s[0]//2+800)],c='r')
    plt.title('Clear high-resolution image')
    cim_clear = im_clear[(s[0]//2-800):(s[0]//2+800),(s[1]//2-800):(s[1]//2+800),0] #grayscale - discard G and B = R
    plt.subplot(122)
    plt.imshow(im_cloudy)
    plt.plot([(s[1]//2-800),(s[1]//2+800)],[(s[0]//2-800),(s[0]//2-800)],c='r')
    plt.plot([(s[1]//2-800),(s[1]//2+800)],[(s[0]//2+800),(s[0]//2+800)],c='r')
    plt.plot([(s[1]//2-800),(s[1]//2-800)],[(s[0]//2-800),(s[0]//2+800)],c='r')
    plt.plot([(s[1]//2+800),(s[1]//2+800)],[(s[0]//2-800),(s[0]//2+800)],c='r')
    plt.title('Cloudy high-resolution image')
    cim_cloudy = im_cloudy[(s[0]//2-800):(s[0]//2+800),(s[1]//2-800):(s[1]//2+800),0] #grayscale - discard G and B =R