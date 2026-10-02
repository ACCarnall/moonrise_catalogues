import numpy as np
import pandas as pd
import os
import logging as log
import warnings

from astropy.table import Table
from astropy import units as u
from astropy.coordinates import SkyCoord

import matplotlib.pyplot as plt

from pair_match_sky import pair_match_sky

make_test_plots = True


# Moonrise catalogue, output from moonrise_uds_merge_catalogues.py
cat = Table.read("moonrise_uds_xmm_catalogue.fits").to_pandas()

# Define necessary masks for setting MOONRISE SF and Q sample flags

mask_good_uvj = ((cat["ABS_U"] > -95) & (cat["ABS_V"] > -95) & (cat["ABS_J"] > -95)
                 & (cat["ABS_U"] < 100) & (cat["ABS_V"] < 100) & (cat["ABS_J"] < 100))

mask_h24 = (cat["HMAG"] <= 24) & (cat["HMAG"] > 0)
mask_h23 = (cat["HMAG"] <= 23) & (cat["HMAG"] > 0)
mask_notstar = cat["STAR"] == 0

mask_zpassive = ((cat["ZBEST"] >= 0.9) & (cat["ZBEST"] <= 1.7)
                 | (cat["ZBEST"] >= 2.0) & (cat["ZBEST"] <= 2.3))

mask_zsf = ((cat["ZBEST"] >= 0.9) & (cat["ZBEST"] <= 1.7)
            | (cat["ZBEST"] >= 2.0) & (cat["ZBEST"] <= 2.6))

mask_uvj_passive = (cat["ABS_U"] - cat["ABS_V"] >= 0.88*(cat["ABS_V"] - cat["ABS_J"]) + 0.49)

# Make combined MOONRISE star-forming and passive masks
moonrise_passive_mask = (mask_h23 & mask_zpassive
                         & mask_uvj_passive & mask_notstar & mask_good_uvj)

moonrise_sf_mask = (mask_h24 & mask_zsf & mask_notstar
                    & ~moonrise_passive_mask & mask_good_uvj)

print(np.sum(moonrise_passive_mask) + np.sum(moonrise_sf_mask))
print(np.sum(mask_h24 & mask_zsf))

cat["in_passive"] = moonrise_passive_mask.astype(int)
cat["in_starforming"] = moonrise_sf_mask.astype(int)

# Set AGN sample flag

cat["in_AGN"] = cat["MOONRISE_ID"].astype(str).str.startswith("13004").astype(int)


# Set high-z sample flag
cat["in_highz"] = np.zeros(len(cat), dtype=int)
cat.loc[cat["HIZ_ID"] > 112, "in_highz"] = 1

Table.from_pandas(cat).write("moonrise_uds_xmm_catalogue_sample_flags.fits",
                             overwrite=True)


# Test plot to check UVJ colours of MOONRISE priorities look sensible
if make_test_plots:
    plt.figure(figsize=(8, 6))
    plt.scatter((cat["ABS_V"] - cat["ABS_J"])[moonrise_sf_mask],
                (cat["ABS_U"] - cat["ABS_V"])[moonrise_sf_mask],
                s=1, alpha=0.1, c="green", label="MOONRISE star-forming")
    plt.scatter((cat["ABS_V"] - cat["ABS_J"])[moonrise_passive_mask],
                (cat["ABS_U"] - cat["ABS_V"])[moonrise_passive_mask],
                s=1, alpha=0.1, c="blue", label="MOONRISE passive")
    plt.xlabel("VJ Colour")
    plt.ylabel("UV Colour")
    plt.legend(frameon=False)
    plt.show()
