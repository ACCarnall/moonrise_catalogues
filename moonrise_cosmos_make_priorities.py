import numpy as np
import pandas as pd
import os
import logging as log
import warnings

from astropy.table import Table
from astropy import units as u
from astropy.coordinates import SkyCoord


# disable INFO level logs from being printed to terminal
log.disable(log.INFO)
log.basicConfig(level=50)

field = "cosmos"

# ##### Load up base tables #####

# COSMOS2020 catalogue, unedited download from
# https://irsa.ipac.caltech.edu/data/COSMOS/tables/cosmos2020/
cat = Table.read("COSMOS2020_CLASSIC_R1_v2.2_p3.fits").to_pandas()

# Cuts to exactly reproduce Ross's 2023 sample selection that was used to
# define objects Bagpipes was run on for UVJ colours and stellar masses
cat = cat[cat["FLAG_COMBINED"] == 0]
cat = cat[cat["UVISTA_H_MAG_APER3"] <= 25]
cat = cat[(cat["ez_z_phot"] >= 0.7) & (cat["ez_z_phot"] <= 2.6)]

print(len(cat))