import numpy as np
import sep
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['figure.figsize'] = [10, 8]

# setting up the visualization parameters

filename = "../data/image.fits"  
hdul = fits.open(NIRCAMimgs)
data = hdul[0].data.astype(np.float32)  
hdul.close()

if data.dtype.byteorder not in ('=', '|'):
    data = data.byteswap().newbyteorder()

# FITS files return data with big-endian byte order, converting to native byte order to avoid errors in SEP functions

m, s = np.mean(data), np.std(data)
plt.imshow(data, cmap='gray', vmin=m-s, vmax=m+s, origin='lower')
plt.colorbar()
plt.title("Original FITS Image")
plt.show()
# setting vmin and vmax based on the mean and standard deviation enhances the contrast


bkg = sep.Background(data)

print("Global background:", bkg.globalback)
print("Global background RMS:", bkg.globalrms)

bkg_image = bkg.back()
plt.imshow(bkg_image, cmap='gray', origin='lower')
plt.colorbar()
plt.title("Background Estimation")
plt.show()

bkg_rms = bkg.rms()
plt.imshow(bkg_rms, cmap='gray', origin='lower')
plt.colorbar()
plt.title("Background Noise")
plt.show()

data_sub = data - bkg

objects = sep.extract(data_sub, thresh=1.5, err=bkg.globalrms)
print("Number of objects detected:", len(objects))
# SEP function returns a structured NumPy array with fields that describe the detected sources

from matplotlib.patches import Ellipse

fig, ax = plt.subplots()
m_sub, s_sub = np.mean(data_sub), np.std(data_sub)
im = ax.imshow(data_sub, cmap='gray', vmin=m_sub-s_sub, vmax=m_sub+s_sub, origin='lower')
plt.colorbar(im, ax=ax)
ax.set_title("Detected Objects Overlay")

for i in range(len(objects)):
    e = Ellipse(xy=(objects['x'][i], objects['y'][i]),
                width=6*objects['a'][i],
                height=6*objects['b'][i],
                angle=objects['theta'][i] * 180. / np.pi,
                edgecolor='red',
                facecolor='none')
    ax.add_artist(e)

plt.show()
# each ellipse represents an object’s size and orientation as determined by the ellipse parameters from SEP

flux, fluxerr, flag = sep.sum_circle(data_sub, objects['x'], objects['y'],
                                     r=3.0, err=bkg.globalrms, gain=1.0)

for i in range(min(10, len(flux))):
    print("Object {:d}: flux = {:f} +/- {:f}".format(i, flux[i], fluxerr[i]))

# photometry helps quantify the brightness of each source


# Calculate mean and standard deviation for scaling the display
m, s = np.mean(data), np.std(data)

plt.imshow(data, cmap='gray', vmin=m-s, vmax=m+s, origin='lower')
plt.colorbar()
plt.title("Original FITS Image")

# Save the figure as a PNG file before showing it
plt.savefig("original_fits_image.png", dpi=300, bbox_inches='tight')
plt.show()


# Evaluate the background as a 2-D array and display it
bkg_image = bkg.back()

plt.imshow(bkg_image, cmap='gray', origin='lower')
plt.colorbar()
plt.title("Background Estimation")

# Save the background estimation figure
plt.savefig("background_estimation.png", dpi=300, bbox_inches='tight')
plt.show()

# Evaluate the background noise as a 2-D array and display it
bkg_rms = bkg.rms()

plt.imshow(bkg_rms, cmap='gray', origin='lower')
plt.colorbar()
plt.title("Background Noise")

# Save the background noise figure 
plt.savefig("background_noise.png", dpi=300, bbox_inches='tight')
plt.show()

from matplotlib.patches import Ellipse

# Plot background-subtracted image
fig, ax = plt.subplots()
m_sub, s_sub = np.mean(data_sub), np.std(data_sub)
im = ax.imshow(data_sub, cmap='gray', vmin=m_sub-s_sub, vmax=m_sub+s_sub, origin='lower')
plt.colorbar(im, ax=ax)
ax.set_title("Detected Objects Overlay")

# Overlay the detected objects as ellipses
for i in range(len(objects)):
    e = Ellipse(xy=(objects['x'][i], objects['y'][i]),
                width=6*objects['a'][i],
                height=6*objects['b'][i],
                angle=objects['theta'][i] * 180. / np.pi,
                edgecolor='red',
                facecolor='none')
    ax.add_artist(e)

# Save the overlay figure before displaying it
plt.savefig("detected_objects_overlay.png", dpi=300, bbox_inches='tight')
plt.show()










