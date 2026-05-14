import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import logging as log
import os

from matplotlib.colors import Normalize

from astropy.table import Table
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
import astropy.units as u

from gaiaunlimited.selectionfunctions import binaries

# disable INFO level logs from being printed to terminal
log.disable(log.INFO)
log.basicConfig(level=50)

np.random.seed(0)


def cutout(ra, dec, mos, size=5.):

    wcs = WCS(mos[0].header)
    wcs.sip = None

    if "CD1_1" in list(mos[0].header):
        cdelt = np.abs(mos[0].header["CD1_1"]*3600.)

    elif "CDELT1" in list(mos[0].header):
        cdelt = np.abs(mos[0].header["CDELT1"]*3600.)

    coord = SkyCoord(ra=ra, dec=dec, unit="deg")

    cutout = Cutout2D(mos[0].data, coord, size/cdelt, wcs=wcs)

    return cutout.data, cdelt


# ##### Load up GAIA star catalogue #####

gaia_table = Table.read("gaia_stars_cosmos.fits").to_pandas()
gaia_table.index = gaia_table["source_id"].astype(int).astype(str).values

# Cut GAIA table by RA and DEC to COSMOS2020 gaia_tablealogue area
ra_mask = (gaia_table["ra"] > 149.3) & (gaia_table["ra"] < 150.79)
dec_mask = (gaia_table["dec"] > 1.6) & (gaia_table["dec"] < 2.82)
gaia_table = gaia_table.groupby(ra_mask & dec_mask).get_group(True)

# Cut by ruwe value to exclude binaries using gaiaunlimited package to
# calculate threshold local to field from Castro-Ginard et al. (2024)
# https://doi.org/10.1051/0004-6361/202450172
sf = binaries.BinarySystemsSelectionFunction()

# Central  coordinates for the COSMOS 2020 catalogue
central_coord = SkyCoord(ra=150.06*u.degree, dec=2.235*u.degree,
                         frame="icrs")

# Returns the RUWE threshold above which a source is considered a
# potential binary, e.g., see Fig. 3 in Castro-Ginard et al. (2024)
ruwe_threshold = sf.query_RUWE(central_coord, crowding=True)

# Define masks to exclude stars we don't want for various reasons
ruwe_mask = (gaia_table["ruwe"] < ruwe_threshold)
single_mask = (gaia_table["non_single_star"] == 0)
pm_mask = (gaia_table["pmra"].abs() < 0.1*1000)
pm_mask = pm_mask & (gaia_table["pmdec"].abs() < 0.1*1000)
var_flag = (gaia_table["phot_variable_flag"] != "VARIABLE")

# combine masks and apply to GAIA table
gaia_star_mask = ruwe_mask & single_mask & pm_mask & var_flag
gaia_table = gaia_table.groupby(gaia_star_mask).get_group(True)


# ##### Load up imaging data #####

# UltraVISTA DR6 J-band mosaic image, unedited download from
# https://irsa.ipac.caltech.edu/data/COSMOS/images/Ultra-Vista/mosaics/
image = "UVISTA_J_12_01_24_allpaw_skysub_015_dr6_rc_v1.fits"
mos = fits.open(image)


# ##### Pick GAIA stars to plot #####

IDs = gaia_table.index.values

indices = np.random.choice(len(IDs), size=100, replace=False)

coords = np.array(gaia_table[["source_id", "ra", "dec"]])

# Not going to change to 2027 epoch here as UltraVISTA imaging isn't new
# coords[:, 1] += gaia_table["pmra"]/1000/3600 * (2027-2016)
# coords[:, 2] += gaia_table["pmdec"]/1000/3600 * (2027-2016)


# ##### Make plot #####

fig = plt.figure(figsize=(10, 12))
gs = mpl.gridspec.GridSpec(10, 10, wspace=0.05, hspace=0.05)

all_axes = []

for i in range(10):
    for j in range(10):
        all_axes.append(plt.subplot(gs[i, j]))

size = 4

all_axes[5].set_title("" + str(size) + "'' \\times " + str(size) + "''")

for i in range(100):

    ID = IDs[indices[i]]
    ax = all_axes[i]

    ra = coords[indices[i], 1]
    dec = coords[indices[i], 2]

    cut, cdelt = cutout(ra, dec, mos, size=size)
    ax.scatter([size/cdelt/2], [size/cdelt/2], s=250, marker="+", lw=0.6,
               color="red", zorder=50)
    ax.imshow(np.flipud(cut), cmap="binary_r",
              norm=Normalize(vmin=np.percentile(cut, 0.5),
                             vmax=np.percentile(cut, 99.5)))

    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.set_xticks([])
    ax.set_yticks([])

for i in range(len(IDs), 100):
    ax = all_axes[i]
    ax.remove()

plt.savefig("astrometry_test_ref_stars_cosmos.pdf", bbox_inches="tight")
plt.close()


# ##### Repeat the process with bright galaxies in cosmos catalogue

cosmos_cat = Table.read("COSMOS2020_CLASSIC_R1_v2.2_p3.fits").to_pandas()

cosmos_cat = cosmos_cat[cosmos_cat["FLAG_UVISTA"] == 0]
zmask = (cosmos_cat["ez_z_phot"] > 0.7) & (cosmos_cat["ez_z_phot"] < 2.6)
cosmos_cat = cosmos_cat[zmask]
mag_mask = ((cosmos_cat["UVISTA_H_MAG_APER3"] < 21)
            & (cosmos_cat["UVISTA_H_MAG_APER3"] > 20))
cosmos_cat = cosmos_cat[mag_mask]

IDs = cosmos_cat["ID"].values

indices = np.random.choice(len(IDs), size=100, replace=False)

coords = np.array(cosmos_cat[["ID", "ALPHA_J2000", "DELTA_J2000"]])


# ##### Make plot #####

fig = plt.figure(figsize=(10, 12))
gs = mpl.gridspec.GridSpec(10, 10, wspace=0.05, hspace=0.05)

all_axes = []

for i in range(10):
    for j in range(10):
        all_axes.append(plt.subplot(gs[i, j]))


all_axes[5].set_title("" + str(size) + "'' \\times " + str(size) + "''")

for i in range(100):

    ID = IDs[indices[i]]
    ax = all_axes[i]

    ra = coords[indices[i], 1]
    dec = coords[indices[i], 2]

    cut, cdelt = cutout(ra, dec, mos, size=size)
    ax.scatter([size/cdelt/2], [size/cdelt/2], s=250, marker="+", lw=0.6,
               color="red", zorder=50)
    ax.imshow(np.flipud(cut), cmap="binary_r",
              norm=Normalize(vmin=np.percentile(cut, 0.5),
                             vmax=np.percentile(cut, 99.5)))

    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.set_xticks([])
    ax.set_yticks([])

for i in range(len(IDs), 100):
    ax = all_axes[i]
    ax.remove()

plt.savefig("astrometry_test_bright_galaxies_cosmos.pdf", bbox_inches="tight")
plt.close()
