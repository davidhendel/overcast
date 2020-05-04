from numpy.lib.stride_tricks import as_strided

def entropy(subim):
    '''Comput the Shannon entropy of an image'''
    p = np.histogram(subim.flatten(),bins=np.linspace(0,255,256), density=True)[0]
    return -sum(p[p>0]*np.log(p[p>0]))

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

clear_filenames=[
'2020-04-02/hires_05h04m11s_2020-04-03_EDT.jpg',
'2020-04-02/hires_05h09m13s_2020-04-03_EDT.jpg',
'2020-04-02/hires_05h14m17s_2020-04-03_EDT.jpg',
'2020-04-02/hires_05h19m20s_2020-04-03_EDT.jpg',
'2020-04-02/hires_05h24m23s_2020-04-03_EDT.jpg',
]

#no moon
nmclear_filenames=[
'2020-04-03/hires_06h54m10s_2020-04-04_EDT.jpg',
'2020-04-03/hires_07h00m33s_2020-04-04_EDT.jpg',
'2020-04-03/hires_07h04m48s_2020-04-04_EDT.jpg',
'2020-04-03/hires_07h10m09s_2020-04-04_EDT.jpg',
'2020-04-03/hires_07h17m36s_2020-04-04_EDT.jpg',
]

mixed_filenames = ['2020-04-05/hires_05h52m59s_2020-04-06_EDT.jpg','2020-04-07/hires_00h27m23s_2020-04-08_EDT.jpg']

#mim = plt.imread('2020-04-05/hires_05h52m59s_2020-04-06_EDT.jpg')[350:1950,1300:2100,0] #800x1600
#im  = plt.imread(clear_filenames[i])[536:1336,1204:2804,0]
#cal = np.resize(np.median(plt.imread(clear_filenames[i])[536:1336,0:500,0],axis=1),(1600,800)).T

artists = []
names = ['Cloudy', 'Clear w/moon', 'Clear w/o moon', 'Mixed']
imsets = [cloudy_filenames, clear_filenames, mixed_filenames]#, nmclear_filenames]
colors = ['r', 'k', 'orange', 'b']





#look for ramps along bottom edge
for n, imset in enumerate(imsets):
    for i in np.arange(len(imset)):
        artist = plt.plot(np.median(plt.imread(imset[i])[-30:],axis=0), c=colors[n], alpha=0.3)[0]
        if i == 0: artists = artists+[artist]
plt.legend(artists, names)

#look for ramps along left and right edge
for n, imset in enumerate(imsets):
    for i in np.arange(len(imset)):
        artist = plt.plot(np.median(plt.imread(imset[i])[:,-250:],axis=1), c=colors[n], alpha=0.3)[0]
        artist = plt.plot(np.median(plt.imread(imset[i])[:,:250],axis=1), c=colors[n], alpha=0.3)[0]
        if i == 0: artists = artists+[artist]
plt.legend(artists, names)



#Calibrated?
#look for ramps along bottom edge
plt.subplot(121)
for n, imset in enumerate(imsets):
    for i in np.arange(len(imset)):
        im = plt.imread(imset[i])[:,:,0]
        cal = np.resize(np.median(im[:,0:500],axis=1),(4008,2672)).T
        im = im-cal
        #plt.plot(np.median(cal[-30:],axis=0), color='g', alpha=0.5, lw=.3)
        artist = plt.plot(np.median(im[-30:],axis=0), c=colors[n], alpha=0.1)[0]
        if i == 0: artists = artists+[artist]
plt.legend(artists, names)
plt.ylabel('median counts')
plt.xlabel('E axis')

#look for ramps along left and right edge
plt.subplot(122)
for n, imset in enumerate(imsets):
    for i in np.arange(len(imset)):
        im = plt.imread(imset[i])[:,:,0]
        cal = np.resize(np.median(im[:,0:500],axis=1),(4008,2672)).T
        im = im-cal
        #plt.plot(np.median(cal[:,-250:],axis=1), color='g', alpha = 0.5, lw=.3)
        artist = plt.plot(np.median(im[:,-250:],axis=1), c=colors[n], alpha=0.3)[0]
        artist = plt.plot(np.median(im[:,:250],axis=1), c=colors[n], alpha=0.3)[0]
        if i == 0: artists = artists+[artist]
plt.legend(artists, names)
plt.ylabel('median counts')
plt.xlabel('N axis')



m_im  = plt.imread(mixed_filenames[1])[:,:,0]
m_cim = m_im[536:1336,1204:2804]
m_cal = np.resize(np.median(m_im[:,0:500],axis=1),(4008,2672)).T
m_calim = (m_im)#/np.median(m_im)#-m_cal)
m_ccalim = m_calim[536:1336,1204:2804]
m_subim  = as_strided(m_ccalim, shape=(800, 1600, 80, 80), strides=m_ccalim.strides+m_ccalim.strides)[::80,::80].reshape(200,80,80)
m_entropyarr = np.zeros(len(m_subim))
for j in np.arange(len(m_subim)):
    m_entropyarr[j] = entropy(m_subim[j])
m_stdarr = np.std(m_subim, axis=(1,2))
m_medianarr = np.median(m_subim, axis=(1,2))

c_im  = plt.imread(clear_filenames[0])[:,:,0]
c_cim = c_im[536:1336,1204:2804]
c_cal = np.resize(np.median(c_im[:,0:500],axis=1),(4008,2672)).T
c_calim = (c_im)#/np.median(c_im) #-c_cal)
c_ccalim = c_calim[536:1336,1204:2804]
c_subim  = as_strided(c_ccalim, shape=(800, 1600, 80, 80), strides=c_ccalim.strides+c_ccalim.strides)[::80,::80].reshape(200,80,80)
c_entropyarr = np.zeros(len(c_subim))
for j in np.arange(len(c_subim)):
    c_entropyarr[j] = entropy(c_subim[j])
c_stdarr = np.std(c_subim, axis=(1,2))
c_medianarr = np.median(c_subim, axis=(1,2))

plt.scatter(c_medianarr, c_stdarr, c='k', s=3)
plt.scatter(m_medianarr, m_stdarr, c='r', s=3)