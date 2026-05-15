import numpy as np
import pandas as pd
import os
import logging as log
import warnings

from astropy.table import Table
from astropy import units as u
from astropy.coordinates import SkyCoord

import matplotlib.pyplot as plt

make_test_plots = True


# ##### Load up base tables #####

# COSMOS2020 catalogue, unedited download from
# https://irsa.ipac.caltech.edu/data/COSMOS/tables/cosmos2020/
cat = Table.read("COSMOS2020_CLASSIC_R1_v2.2_p3.fits").to_pandas()

cat.rename(columns={"ALPHA_J2000": "ra",
                    "DELTA_J2000": "dec"}, inplace=True)

# Cuts to exactly reproduce Ross's 2023 sample selection that was used to
# define objects Bagpipes was run on for UVJ colours and stellar masses
# cat = cat[cat["FLAG_COMBINED"] == 0]
# cat = cat[cat["UVISTA_H_MAG_APER3"] <= 25]
# cat = cat[(cat["ez_z_phot"] >= 0.7) & (cat["ez_z_phot"] <= 2.6)]

# Bagpipes fit results for COSMOS2020 sub-sample
pipes_cat = Table.read("cosmos_bagpipes_subsample.fits").to_pandas()

cols_to_drop = cat.columns[np.isin(cat.columns, pipes_cat.columns)]
pipes_cat.drop(columns=cols_to_drop, inplace=True)
pipes_cat.rename(columns={"id": "ID"}, inplace=True)

# Merge COSMOS2020 catalogue with Bagpipes fit results catalogue
cat = pd.merge(cat, pipes_cat, left_on="ID", right_on="ID", how="left",
               suffixes=('_1', '_2'))

# Define necessary masks for setting MOONRISE priorities
mask_phot_combined = cat["FLAG_COMBINED"] == 0

mask_h24 = cat["UVISTA_H_MAG_APER3"] <= 24
mask_h23 = cat["UVISTA_H_MAG_APER3"] <= 23

mask_zpassive = ((cat["ez_z_phot"] >= 0.7) & (cat["ez_z_phot"] <= 1.7)
                 | (cat["ez_z_phot"] >= 2.0) & (cat["ez_z_phot"] <= 2.3))

mask_zsf = ((cat["ez_z_phot"] >= 0.7) & (cat["ez_z_phot"] <= 1.7)
            | (cat["ez_z_phot"] >= 2.0) & (cat["ez_z_phot"] <= 2.6))

mask_uvj_passive = (cat["UV_colour_50"] >= 0.88*cat["VJ_colour_50"] + 0.49)

# Make combined MOONRISE star-forming and passive masks
moonrise_passive_mask = (mask_h23 & mask_zpassive & mask_phot_combined
                         & mask_uvj_passive)

moonrise_sf_mask = (mask_h24 & mask_zsf & mask_phot_combined
                    & ~moonrise_passive_mask)

print(np.sum(moonrise_passive_mask) + np.sum(moonrise_sf_mask))
print(np.sum(mask_h24 & mask_zsf & mask_phot_combined))

# Test plot to check UVJ colours of MOONRISE priorities look sensible
if make_test_plots:
    plt.figure(figsize=(8, 6))
    plt.scatter(cat["VJ_colour_50"][moonrise_sf_mask],
                cat["UV_colour_50"][moonrise_sf_mask],
                s=1, c="green", label="MOONRISE star-forming")
    plt.scatter(cat["VJ_colour_50"][moonrise_passive_mask],
                cat["UV_colour_50"][moonrise_passive_mask],
                s=1, c="blue", label="MOONRISE passive")
    plt.xlabel("VJ Colour")
    plt.ylabel("UV Colour")
    plt.legend(frameon=False)
    plt.show()
