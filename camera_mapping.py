import numpy as np
import datetime
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
import emcee
import corner
_DIR = './'

NMSkies = EarthLocation(lat=32.90233493*u.deg, lon=-105.5309338*u.deg, height=2225*u.m)
utcoffset = -4*u.hour #(EDT, system time when downloaded)

#Full star catalog
ybsc = np.genfromtxt(_DIR+'yale-bright-star-catalog.txt', 
                        skip_header=45, usecols=(0,1,9), loose=True, invalid_raise=False)
ybscSkyCoord = SkyCoord(ra=ybsc[:,0]*u.deg, dec=ybsc[:,1]*u.deg)
ybscmags = ybsc[:,2]

def time_from_fname(fname, utcoffset):
    #Return datetime object of observation given the filename
    #file naming scheme could be better
    format_time = (fname.split('_')[1]+'_'+fname.split('_')[2]+'_'+fname.split('_')[3]).split('.')[0]
    obstime = Time(datetime.datetime.strptime(format_time,'%Hh%Mm%Ss_%Y-%m-%d_%Z')) - utcoffset
    return obstime
    
def visible_stars_altaz(obstime, location=NMSkies, cat=ybscSkyCoord, mags=ybscmags, minmag=4):
    #convert RA, Dec to local NMSkies altaz coordinates; return those above the horizon
    altaz = cat.transform_to(AltAz(obstime=obstime,location=location))
    return altaz[((altaz.alt>0)&(mags<minmag))]

def fisheye_mapping(altaz, scalefac=2142, rot = 0., k2=2, xc = 2004, yc = 1336):
    '''
    Purpose: map sky position to image coordinates
    
    Input: an Astropy SkyCoord in the AltAZ frame; optional camera parameters
    
    Returns: x and y pixel positions of stars given the camera parameters
    
    f=5.9 is for circular fisheyes on APS-C detectors with r = 8.4mm
    Combine f, k1 and pixel scaling into scalefac
    Rot is extra rotation East of North in deg
    equisolid mapping is r = 2*scalefac*sin(theta/k2), theta in rad with pi/2 as the horizon
    North (az=0 deg) is towards +y, East (az=90 deg) is towards -x
    '''
    r = scalefac*np.sin(-(np.pi/2.-altaz.alt.to(u.rad).value)/k2)
    x = r*np.sin((altaz.az+rot*u.deg).to(u.rad).value)+xc
    y = r*np.cos((altaz.az+rot*u.deg).to(u.rad).value)+yc
    return x, y

#Use this image to work out the mapping between sky and pixels
imfname = _DIR+'2020-04-02/hires_05h44m35s_2020-04-03_EDT.jpg'
im = plt.imread(imfname)
obstime = time_from_fname(imfname, utcoffset)

#Bright stars in a few constellations with pixel coordinates identified
catfname = _DIR+'ybs-mapped.txt'
catalog = np.genfromtxt(catfname,skip_header=1, usecols=(0,1,11,12,13), invalid_raise=False)
catSkyCoord = SkyCoord(ra=catalog[:,0]*u.deg, dec=catalog[:,1]*u.deg)
catMags = catalog[:,2]
catX = catalog[:,3]
catY = catalog[:,4]
catErr = np.ones(len(catX))*3. #assume we can measure the center of a star within a few pixels

catAltaz = visible_stars_altaz(obstime, location=NMSkies, cat=catSkyCoord, mags=catMags, minmag=6)


#theta = parameters = (scalefac, rot, k2, xc, yc)
#take wide Gaussian priors on scalefac, rot, xc, and yc; flat priors on k1 and k2
#scalefac depends on camera and detector pixel spacing, ~900 by inspection
#k1 and k2 are parameters of the lens, should be ~2

mus    = np.array([2142., 0., 2004., 1336.])
sigmas = np.array([300., 10., 100., 100.])

def log_likelihood(theta, catX, catY, catErr):
    #take the likelihood as the Gaussian distance between image and catalog x,y values
    scalefac, rot, k2, xc, yc = theta
    x, y = fisheye_mapping(catAltaz, scalefac=scalefac, rot=rot, k2=k2, xc=xc, yc=yc)
    dists = np.sqrt((x-catX)**2+(y-catY)**2)
    return -0.5*np.sum(((dists)/catErr)**2)

def log_prior(theta, mus = mus, sigmas = sigmas):
    #wide Gaussian priors on scalefac, rot, xc, and yc; flat priors on k2
    scalefac, rot, k2, xc, yc = theta
    if not ((1 < k2 < 3)):# and (1 < k1 < 3)):
        return -np.inf
    else:
        return -0.5*np.sum((np.array([scalefac,rot,xc,yc])-mus)**2/sigmas**2)

def log_probability(theta, x, y, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr)

def run_emcee():
    #initalize walker start
    walker_init_pos              = np.zeros((48,5))
    walker_init_pos[:,[0,1,3,4]] = np.random.randn(48,4)*sigmas+mus
    walker_init_pos[:,[2]]     = np.random.uniform(low=1,high=3,size=(48,1))
    nwalkers, ndim = walker_init_pos.shape
    #run MCMC with emcee
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(catX, catY, catErr))
    sampler.run_mcmc(walker_init_pos, 10000)
    return sampler

def plot_corner(sampler, save=False):
    afterburn = sampler.chain[:,2000:,:].reshape((384000,5))
    corner.corner(afterburn, labels = ['Scaling', 'Rotation', 'k2', 'x center', 'y center'],
                 show_titles=True, quantiles=(.14,.50,.84),title_kwargs={"fontsize": 10})
    fig = plt.gcf()
    fig.set_size_inches(12,12)
    if save: plt.savefig(_DIR+'corner_plot_2.png',dpi=200, bbox_inches='tight')
    return

        
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#check that stars on image match measurements correctly
#seems ok
def star_pos_check(scalefac = 900*2, rot = 0., k2=2, xc = 2004, yc = 1336):
    plt.figure(figsize=(8,4))
    plt.clf()
    x,y=fisheye_mapping(catAltaz, scalefac=scalefac, rot=rot, k2=k2, xc=xc, yc=yc)
    #Scorpio
    plt.subplot(131)
    plt.imshow(im)
    plt.title('Scorpius')
    plt.plot(x, y, linestyle='', marker = 's', markersize=7, 
                 markerfacecolor='none', markeredgecolor='b', label = 'from YBSC')
    plt.plot(catX, catY, linestyle='', marker = 'o', markersize=5, 
             markerfacecolor='none', markeredgecolor='r' )
    plt.ylim(2500,2100)
    plt.xlim(1600,2000)
    #Big Dipper
    plt.subplot(132)
    plt.imshow(im)
    plt.title('Ursa Major')
    plt.plot(x, y, linestyle='', marker = 's', markersize=7, 
                 markerfacecolor='none', markeredgecolor='b', label = 'from YBSC')
    plt.plot(catX, catY, linestyle='', marker = 'o', markersize=5, 
             markerfacecolor='none', markeredgecolor='r' )
    plt.xlim(2000,2800)
    plt.ylim(1200,400)
    #Lyra
    plt.subplot(133)
    plt.imshow(im)
    plt.title('Lyra')
    plt.plot(x, y, linestyle='', marker = 's', markersize=7, 
                 markerfacecolor='none', markeredgecolor='b', label = 'From catalog')
    plt.plot(catX, catY, linestyle='', marker = 'o', markersize=5, 
             markerfacecolor='none', markeredgecolor='r', label = 'From image')
    plt.xlim(1250,1450)
    plt.ylim(1300,1100)
    plt.legend(loc='lower right', fontsize='small')
    fig = plt.gcf()
    #fig.title('Assuming camera is pointed exactly at zenith and pointed true North')
    return

#check transformation function
if 0:
    plt.clf()
    plt.imshow(im)
    x, y = fisheye_mapping(catAltaz)
    plt.plot(x, y, linestyle='', marker = 's', markersize=7, 
                 markerfacecolor='none', markeredgecolor='b', label = 'from YBSC')
    plt.plot(catX, catY, linestyle='', marker = 'o', markersize=5, 
                 markerfacecolor='none', markeredgecolor='r', label = 'from image')
    plt.legend()
