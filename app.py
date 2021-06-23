import numpy 
import matplotlib.pyplot as plt 
import streamlit as st 
import matplotlib.colors as colors
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
from scipy.fftpack import fftshift, ifftshift, fft2, ifft2
import base64
import datetime
import io
import base64
from io import BytesIO
import PIL
from scipy import ndimage

st.title('PHI/HRT telescope tool!')

######################## Functions

def Zernike_polar(coefficients, r, u, co_num):
 Z =  np.zeros(37)
 Z[:co_num] = coefficients
 #Z1  =  Z[0]  * 1*(np.cos(u)**2+np.sin(u)**2)
 #Z2  =  Z[1]  * 2*r*np.cos(u)
 #Z3  =  Z[2]  * 2*r*np.sin(u)
 
 Z4  =  Z[0]  * np.sqrt(3)*(2*r**2-1)  #defocus

 Z5  =  Z[1]  * np.sqrt(6)*r**2*np.sin(2*u) #astigma
 Z6  =  Z[2]  * np.sqrt(6)*r**2*np.cos(2*u)
 
 Z7  =  Z[3]  * np.sqrt(8)*(3*r**2-2)*r*np.sin(u) #coma
 Z8  =  Z[4]  * np.sqrt(8)*(3*r**2-2)*r*np.cos(u)
 
 Z9  =  Z[5]  * np.sqrt(8)*r**3*np.sin(3*u) #trefoil
 
 Z10=  Z[6] * np.sqrt(8)*r**3*np.cos(3*u)
 
 Z11 =  Z[7] * np.sqrt(5)*(1-6*r**2+6*r**4) #secondary spherical
 
 Z12 =  Z[8] * np.sqrt(10)*(4*r**2-3)*r**2*np.cos(2*u)  #2 astigma
 Z13 =  Z[9] * np.sqrt(10)*(4*r**2-3)*r**2*np.sin(2*u)
 
 Z14 =  Z[10] * np.sqrt(10)*r**4*np.cos(4*u) #tetrafoil
 Z15 =  Z[11] * np.sqrt(10)*r**4*np.sin(4*u)
 
 Z16 =  Z[12] * np.sqrt(12)*(10*r**4-12*r**2+3)*r*np.cos(u) #secondary coma
 Z17 =  Z[13] * np.sqrt(12)*(10*r**4-12*r**2+3)*r*np.sin(u)
 
 Z18 =  Z[14] * np.sqrt(12)*(5*r**2-4)*r**3*np.cos(3*u) #secondary trefoil
 Z19 =  Z[15] * np.sqrt(12)*(5*r**2-4)*r**3*np.sin(3*u)

 Z20 =  Z[16] * np.sqrt(12)*r**5*np.cos(5*u) #pentafoil
 Z21 =  Z[17] * np.sqrt(12)*r**5*np.sin(5*u)
 
 Z22 =  Z[18] * np.sqrt(7)*(20*r**6-30*r**4+12*r**2-1) #spherical
 
 Z23 =  Z[19] * np.sqrt(14)*(15*r**4-20*r**2+6)*r**2*np.sin(2*u) #astigma
 Z24 =  Z[20] * np.sqrt(14)*(15*r**4-20*r**2+6)*r**2*np.cos(2*u)
 
 Z25 =  Z[21] * np.sqrt(14)*(6*r**2-5)*r**4*np.sin(4*u)#trefoil
 Z26 =  Z[22] * np.sqrt(14)*(6*r**2-5)*r**4*np.cos(4*u)
 
 Z27 =  Z[23] * np.sqrt(14)*r**6*np.sin(6*u) #hexafoil 
 Z28 =  Z[24] * np.sqrt(14)*r**6*np.cos(6*u)

 Z29 =  Z[25] * 4*(35*r**6-60*r**4+30*r**2-4)*r*np.sin(u) #coma
 Z30 =  Z[26] * 4*(35*r**6-60*r**4+30*r**2-4)*r*np.cos(u)
 
 Z31 =  Z[27] * 4*(21*r**4-30*r**2+10)*r**3*np.sin(3*u)#trefoil
 Z32 =  Z[28] * 4*(21*r**4-30*r**2+10)*r**3*np.cos(3*u)

 Z33 =  Z[29] * 4*(7*r**2-6)*r**5*np.sin(5*u) #pentafoil
 Z34 =  Z[30] * 4*(7*r**2-6)*r**5*np.cos(5*u)
 
 Z35 =  Z[31] * 4*r**7*np.sin(7*u) #heptafoil
 Z36 =  Z[32] * 4*r**7*np.cos(7*u)
 
 Z37 =  Z[33] * 3*(70*r**8-140*r**6+90*r**4-20*r**2+1) #spherical
 
#Z1+Z2+Z3+
 ZW = Z4+Z5+Z6+Z7+Z8+Z9+Z10+Z11+Z12+Z13+Z14+Z15+Z16+ Z17+Z18+Z19+Z20+Z21+Z22+Z23+ Z24+Z25+Z26+Z27+Z28+ Z29+ Z30+ Z31+ Z32+ Z33+ Z34+ Z35+ Z36+ Z37
 return ZW


def spat_res(la,D):
        return 206265*lam/D

def pupil_size(D,lam,pix,size):
        pixrad = pix*np.pi/(180*3600)  # Pixel-size in radians
        nu_cutoff = D/lam      # Cutoff frequency in rad^-1
        deltanu = 1./(size*pixrad)     # Sampling interval in rad^-1
        rpupil = nu_cutoff/(2*deltanu) #pupil size in pixels
        return np.int(rpupil)

def arctokm(d):
        R = 696000 #in km
        D = d*1.49598e8#149600000 #km
        alpha_rad = 2*np.arctan(R/(D)) #size of Sun's diameter in rad
        alpha_arc = alpha_rad*206265
        
        factor = (2*R)/alpha_arc
        return alpha_arc, factor

## function for making the phase in a unit circle
def phase(coefficients,rpupil,co_num):
 r = 1
 x = np.linspace(-r, r, 2*rpupil)
 y = np.linspace(-r, r, 2*rpupil)

 [X,Y] = np.meshgrid(x,y) 
 R = np.sqrt(X**2+Y**2)
 theta = np.arctan2(Y, X)
    
 Z = Zernike_polar(coefficients,R,theta,co_num)
 Z[R>1] = 0
 return Z


def mask(rpupil, size):
 r = 1
 x = np.linspace(-r, r, 2*rpupil)
 y = np.linspace(-r, r, 2*rpupil) 

 [X,Y] = np.meshgrid(x,y) 
 R = np.sqrt(X**2+Y**2)
 theta = np.arctan2(Y, X)
 M = 1*(np.cos(theta)**2+np.sin(theta)**2)
 M[R>1] = 0
 Mask =  np.zeros([size,size])
 Mask[size//2-rpupil+1:size//2+rpupil+1,size//2-rpupil+1:size//2+rpupil+1]= M
 return Mask

def PSF(complx_pupil):
    PSF = ifftshift(fft2(fftshift(complx_pupil))) 
    PSF = (np.abs(PSF))**2 #or PSF*PSF.conjugate()
    PSF = PSF/PSF.sum() #normalizing the PSF
    return PSF


## function to compute the OTF from PSF (to be used in PD fit )
def OTF(psf):
    otf = ifftshift(psf)
    otf = fft2(otf)
    otf = otf/float(otf[0,0])
    #sotf = otf/otf.max() # or otf_max = otf[size/2,size/2] if max is shifted to center
   
    return otf


def MTF(otf):
    mtf = np.abs(otf)
    return mtf


def center(coefficients,size,rpupil,co_num):
    A = np.zeros([size,size])
    A[size//2-rpupil+1:size//2+rpupil+1,size//2-rpupil+1:size//2+rpupil+1]= phase(coefficients,rpupil,co_num)
    return A

def complex_pupil(A,Mask):
    abbe =  np.exp(1j*A)
    abbe_z = np.zeros((len(abbe),len(abbe)),dtype=np.complex)
    abbe_z = Mask*abbe
    return abbe_z

def figure(data):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1,1,1)
    im  = ax.imshow(data,cmap=plt.get_cmap('inferno'),origin='lower')
    #ax.set_xlabel('PIXELS',fontsize=20)
    #ax.set_ylabel('PIXELS',fontsize=20)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=0.15, pad=0.05)
    cbar = plt.colorbar(im, cax=cax,orientation='vertical')
    return fig
    '''
def get_image_download_link(img):
    """Generates a link allowing the PIL image to be downloaded
    in:  PIL image
    out: href string
    """
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/jpg;base64,{img_str}">Download result</a>'
    return href
    '''
def GetPSD1D(psd2D):
    h  = psd2D.shape[0]
    w  = psd2D.shape[1]
    wc = w//2
    hc = h//2

    # create an array of integer radial distances from the center
    Y, X = np.ogrid[0:h, 0:w]
    r    = np.hypot(X - wc, Y - hc).astype(np.int)

    # SUM all psd2D pixels with label 'r' for 0<=r<=wc
    # NOTE: this will miss power contributions in 'corners' r>wc
    psd1D = ndimage.mean(psd2D, r, index=np.arange(0, wc))

    return psd1D

def plot_az(mtf_2d,d,lam,f):
    az = GetPSD1D(mtf_2d)
    freq=np.linspace(0,0.5,int(mtf_2d.shape[0]/2))
    freq_c_hrt = d/(lam*f*100)
    phi_hrt = np.arccos(freq/freq_c_hrt)
    MTF_p_hrt = (2/np.pi)*(phi_hrt - (np.cos(phi_hrt))*np.sin(phi_hrt))
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('Freq (1/pixel)',fontsize=22)
    ax.set_ylabel('MTF',fontsize=22)
    ax.plot(freq,MTF_p_hrt,label='Theoretical MTF')
    ax.plot(freq,az,label='Observed MTF')
    ax.set_xlim(0,0.5)
    plt.legend()
    return fig



menu = ['Figure out my telescope', 'Work with aberrations']
choice = st.sidebar.selectbox('What do you want to do?',menu)
#st.subheader('test')
ap = st.sidebar.number_input('Telescope aperture size (in nm):',value=140)
st.write('You selected a telescope with an entrance aperture of ', ap, 'nm')

lam = st.sidebar.number_input('Telescope working wavelength (in nm):',value=617.3*10**(-6),min_value=1e-7, max_value=0.1)
st.write('You selected a working wavelength of', lam, 'nm')


focal = st.sidebar.number_input('Effective focal length (in nm):',value=4125.3)
st.write('You have selected a telescope with an effective focal length of', focal, 'nm')

pix_size = st.sidebar.number_input('Pixel size (in arcseconds):',value=0.5)
st.write('You have selected a pixel size of', pix_size, 'arcseconds/pixel')

size = st.sidebar.number_input('Size of the detector (in pixels):',value=2048)
st.write('You have selected a camera with size of', size, 'pixels')

distance = st.sidebar.number_input('Distance of Solar Orbiter to the Sun (in AU):',value=0.5) 
st.write('You have selected SOLO-SUN distance of', distance, 'AU')
options = st.selectbox('What would you like to compute?',['spatial resolution (in arcsec)','spatial resolution (in km)', 'pupil size'])

if choice == 'Figure out my telescope':
    if options == 'spatial resolution (in arcsec)':
        st.write('The spatial resolution of your optical setup is', spat_res(lam,ap),'arcsec' )
    elif options == 'pupil size':
        st.write('The pupil size of your optical setup is', pupil_size(ap,lam,pix_size,size),'pixels' )
    elif options == 'spatial resolution (in km)':
        st.write('The HRT resolution at', distance, 'AU', 'is', 0.5*arctokm(distance)[1], 'km')

    
if choice == 'Work with aberrations':
    st.sidebar.title('Choose Zernike coefficients number:')
    z = st.sidebar.selectbox("Number of Zernike Polynomials",[2,3,4])
    coefficients = []
    for i in np.arange(z):
        val = st.sidebar.number_input('Value of Zernike coefficient #'+str(i)+':')
        coefficients.append(val)
    coefficients = np.asarray(coefficients)
    st.write('You selected the first',z, 'Zernike Polynomials')
    rpupil = pupil_size(ap,lam,pix_size,size)
    sim_phase = center(coefficients,size,rpupil,z)
    Mask = mask(rpupil, size)
    pupil_com = complex_pupil(sim_phase,Mask)
    psf = PSF(pupil_com)
    otf = OTF(psf)
    mtf = MTF(otf)
    options3 = st.selectbox('What do you want to plot?',['Wavefront', '2D PSF', '2D MTF','1D MTF'])
    if options3 == 'Wavefront':
        st.pyplot(figure(sim_phase))        
    elif options3 == '2D PSF':
        st.pyplot(figure(np.log(np.abs(psf))))
    elif options3 == '2D MTF':
        st.pyplot(figure(fftshift(mtf)))
    elif options3 == '1D MTF':
        st.pyplot(plot_az(fftshift(mtf),ap,lam,focal))
    
