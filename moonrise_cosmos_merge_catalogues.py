import numpy as np
import pandas as pd
import os
import logging as log
import warnings

from astropy.table import Table
from astropy import units as u
from astropy.coordinates import SkyCoord

from gaiaunlimited.selectionfunctions import binaries


# ##### Load up base tables #####

# COSMOS2020 catalogue, unedited download from
# https://irsa.ipac.caltech.edu/data/COSMOS/tables/cosmos2020/
cosmos2020 = Table.read("COSMOS2020_CLASSIC_R1_v2.2_p3.fits").to_pandas()

# GAIA star catalogue, unedited download using only ra/dec criteria from
# https://gea.esac.esa.int/archive/, SQL query text saved with this file
gaia_table = Table.read(f"gaia_stars_{field}.fits").to_pandas()


# ##### Apply cuts to GAIA star catalogue #####

# Cut GAIA table by RA and DEC to COSMOS2020 catalogue area
ra_mask = (gaia_table["ra"] > 149.05) & (gaia_table["ra"] < 151.07)
dec_mask = (gaia_table["dec"] > 1.39) & (gaia_table["dec"] < 3.08)
gaia_table = gaia_table[ra_mask & dec_mask]

# Cut by ruwe value to exclude binaries using gaiaunlimited package to
# calculate threshold local to field from Castro-Ginard et al. (2024)
# https://doi.org/10.1051/0004-6361/202450172
sf = binaries.BinarySystemsSelectionFunction()

# Central  coordinates for the COSMOS 2020 catalogue
central_coord = SkyCoord(ra=150.06*u.degree, dec=2.235*u.degree, frame="icrs")

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
gaia_table = gaia_table[gaia_star_mask]

# Change from default GAIA DR3 epoch of 2016 to 2027 for MOONRISE
gaia_table["ra"] += gaia_table["pmra"]/1000/3600*(2027-2016)
gaia_table["dec"] += gaia_table["pmdec"]/1000/3600*(2027-2016)


# ##### Merge GAIA star catalogue into main catalogue #####


# ##### Sort out best redshift column #####


# ##### Merge in bagpipes results catalogue #####
# Cuts to exactly reproduce Ross's 2023 sample selection that was used to
# define objects Bagpipes was run on for UVJ colours and stellar masses
# cat = cat[cat["FLAG_COMBINED"] == 0]
# cat = cat[cat["UVISTA_H_MAG_APER3"] <= 25]
# cat = cat[(cat["ez_z_phot"] >= 0.7) & (cat["ez_z_phot"] <= 2.6)]

# Bagpipes fit results for COSMOS2020 sub-sample
pipes_cat = Table.read("cosmos_bagpipes_subsample.fits").to_pandas()

cols_to_drop = cosmos2020.columns[np.isin(cosmos2020.columns,
                                          pipes_cat.columns)]
pipes_cat.drop(columns=cols_to_drop, inplace=True)
pipes_cat.rename(columns={"id": "ID"}, inplace=True)

# Merge COSMOS2020 catalogue with Bagpipes fit results catalogue
cosmos2020 = pd.merge(cosmos2020, pipes_cat, left_on="ID", right_on="ID",
                      suffixes=('_1', '_2'))


# ##### Merge in high-z and AGN (and other?) source catalogues #####
