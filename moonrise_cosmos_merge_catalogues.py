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

cosmos2020["PMRA"] = 0.
cosmos2020["PMDEC"] = 0.
cosmos2020.rename(columns={"ALPHA_J2000": "RA",
                            "DELTA_J2000": "DEC",
                            "ez_z_phot": "ZPHOT"}, inplace=True)

cosmos2020["MAG"] = -99.
cosmos2020.loc[cosmos2020["UVISTA_H_MAG_APER3"] > 0, "MAG"] = cosmos2020["UVISTA_H_MAG_APER3"]

# Estimate SIZE in arcsec(defined as diameter containing ~all of the flux)
# from FLUX_RADIUS, which is the half-light radius in pixels (0.15"/pixel)

cosmos2020["SIZE"] = -99.
cosmos2020.loc[cosmos2020["FLUX_RADIUS"] > 0, "SIZE"] = 4*cosmos2020["FLUX_RADIUS"]*0.15

cosmos2020["STAR"] = 0

# Khostovan et al. (2025) v1.1 specz compilation, unedited download from
# https://github.com/cosmosastro/speczcompilation
specz_path = "specz_compilation_COSMOS_DR1.1_unique.fits"
khost = Table.read(specz_path).to_pandas()
khost.rename(columns={"specz": "ZSPEC_KHOSTOVAN", "flag": "ZFLAG_KHOSTOVAN"},
             inplace=True)

# Cut Khostovan catalogue to objects that have counterparts in cosmos2020
# catalogue, as well as high confidence speczs (flags 3 and 4)
khost = khost[khost["Id_COS20_Classic"] > 0]
khost = khost[khost["Confidence_level"] >= 95]

khost = khost[["Id_COS20_Classic", "ZSPEC_KHOSTOVAN", "ZFLAG_KHOSTOVAN"]]

# Match using cosmos2020 and cosmos2025 IDs provided by Khostovan et al.
cosmos2020 = pd.merge(cosmos2020, khost, how="outer", left_on="ID", right_on="Id_COS20_Classic")

cosmos2020.drop(columns=["Id_COS20_Classic"], inplace=True)

# GAIA star catalogue, unedited download using only ra/dec criteria from
# https://gea.esac.esa.int/archive/, SQL query text saved with this file
gaia_table = Table.read("gaia_stars_cosmos.fits").to_pandas()

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

# Set the STAR flag to 1 for all gaia objects
gaia_table["STAR"] = 1

# Merge in 2mass H-band magnitudes for GAIA stars


# ##### Merge GAIA star catalogue into main catalogue #####

gaia_table = gaia_table[["source_id", "ra", "dec", "pmra", "pmdec", "ruwe",
                         "STAR", "phot_rp_mean_mag"]]

gaia_table.rename(columns={"source_id": "GAIA_STAR_ID", "ra": "RA", "dec": "DEC",
                           "pmra": "PMRA", "pmdec": "PMDEC", "ruwe": "RUWE",
                           "phot_rp_mean_mag": "GAIA_magR"}, inplace=True)

cosmos2020 = pd.concat([cosmos2020, gaia_table], ignore_index=True)

# ##### Merge in bagpipes results catalogue #####

# Cuts to exactly reproduce Ross's 2023 sample selection that was used to
# define objects Bagpipes was run on for UVJ colours and stellar masses
# cat = cat[cat["FLAG_COMBINED"] == 0]
# cat = cat[cat["UVISTA_H_MAG_APER3"] <= 25]
# cat = cat[(cat["ez_z_phot"] >= 0.7) & (cat["ez_z_phot"] <= 2.6)]

# Bagpipes fit results for COSMOS2020 sub-sample
pipes_cat = Table.read("cosmos_bagpipes_subsample.fits").to_pandas()

pipes_cat = pipes_cat[["id", "stellar_mass_16", "stellar_mass_50",
                       "stellar_mass_84", "U_50", "V_50", "J_50"]]

pipes_cat.rename(columns={"id": "ID", "U_50": "ABS_U", "V_50": "ABS_V",
                          "J_50": "ABS_J"}, inplace=True)

# Merge COSMOS2020 catalogue with Bagpipes fit results catalogue
cosmos2020 = pd.merge(cosmos2020, pipes_cat, how="outer", on=None)


# ##### Merge in high-z and AGN (and other?) source catalogues #####


# ##### Sort out best redshift column #####

cosmos2020["ZBEST"] = -99.
cosmos2020.loc[cosmos2020["ZPHOT"] > 0, "ZBEST"] = cosmos2020["ZPHOT"]
cosmos2020.loc[cosmos2020["ZSPEC_KHOSTOVAN"] > 0, "ZBEST"] = cosmos2020["ZSPEC_KHOSTOVAN"]


# ##### Keep only necessary columns, set -99s and save to file #####

cosmos2020.fillna(dict(zip(cosmos2020.columns, [-99] * cosmos2020.shape[0])),
                   inplace=True)
cosmos2020.fillna({"ZFLAG_KHOSTOVAN": -99}, inplace=True)

cosmos2020 = cosmos2020[["ID", "RA", "DEC", "PMRA", "PMDEC", "MAG", "SIZE",
                         "FLAG_COMBINED", "ZBEST", "ZPHOT", "ZSPEC_KHOSTOVAN",
                         "ZFLAG_KHOSTOVAN", "STAR", "GAIA_STAR_ID",
                         "GAIA_magR", "RUWE", "ABS_U", "ABS_V", "ABS_J",
                         "stellar_mass_16", "stellar_mass_50", "stellar_mass_84"]]

Table.from_pandas(cosmos2020).write("moonrise_cosmos_catalogue.fits",
                                    overwrite=True)