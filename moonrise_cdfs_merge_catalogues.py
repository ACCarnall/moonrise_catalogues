import numpy as np
import pandas as pd
import os
import logging as log
import warnings

from astropy.table import Table
from astropy import units as u
from astropy.coordinates import SkyCoord

from gaiaunlimited.selectionfunctions import binaries

from pair_match_sky import pair_match_sky

# ##### Load up base tables #####

# derek_cdfs catalogue, unedited download from
# https://irsa.ipac.caltech.edu/data/COSMOS/tables/derek_cdfs/
derek_cdfs = Table.read("CDFS1+2+3_Hband_3sigma_z0p0_z4p0_101122.fits").to_pandas()

derek_cdfs["PMRA"] = 0.
derek_cdfs["PMDEC"] = 0.
derek_cdfs.rename(columns={"zphot": "ZPHOT",
                           "ID": "derek_cdfs_ID"}, inplace=True)

derek_cdfs["HMAG"] = 23.9-2.5*np.log10(derek_cdfs["VIDEO_H"])
derek_cdfs["HMAG_FLAG"] = 0

derek_cdfs["SIZE"] = -99.

derek_cdfs["STAR"] = 0

derek_cdfs["MOONRISE_ID"] = derek_cdfs["derek_cdfs_ID"].astype(str)
derek_cdfs["MOONRISE_ID"] = "12001" + derek_cdfs["MOONRISE_ID"].str.zfill(9)

# Bagpipes fit results for COSMOS2020 sub-sample
pipes_cat = Table.read("ecdfs_bagpipes_subsample.fits").to_pandas()

pipes_cat = pipes_cat[["id", "stellar_mass_16", "stellar_mass_50",
                       "stellar_mass_84", "U_50", "V_50", "J_50"]]

pipes_cat.rename(columns={"id": "ID", "U_50": "ABS_U", "V_50": "ABS_V",
                          "J_50": "ABS_J"}, inplace=True)

# Merge COSMOS2020 catalogue with Bagpipes fit results catalogue
derek_cdfs = pd.merge(derek_cdfs, pipes_cat, how="outer",
                      left_on="derek_cdfs_ID", right_on="ID")