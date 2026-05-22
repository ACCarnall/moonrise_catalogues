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

# COSMOS2020 catalogue, unedited download from
# https://irsa.ipac.caltech.edu/data/COSMOS/tables/cosmos2020/
cosmos2020 = Table.read("COSMOS2020_CLASSIC_R1_v2.2_p3.fits").to_pandas()

cosmos2020["PMRA"] = 0.
cosmos2020["PMDEC"] = 0.
cosmos2020.rename(columns={"ALPHA_J2000": "RA",
                           "DELTA_J2000": "DEC",
                           "ez_z_phot": "ZPHOT",
                           "ID": "COSMOS2020_ID"}, inplace=True)

cosmos2020["HMAG"] = -99.
cosmos2020["HMAG_FLAG"] = -99.
H_mask = (cosmos2020["UVISTA_H_MAG_APER3"] > 0)
cosmos2020.loc[H_mask, "HMAG"] = cosmos2020.loc[H_mask, "UVISTA_H_MAG_APER3"]
cosmos2020.loc[H_mask, "HMAG_FLAG"] = 0

# Estimate SIZE in arcsec(defined as diameter containing ~all of the flux)
# from FLUX_RADIUS, which is the half-light radius in pixels (0.15"/pixel)

cosmos2020["SIZE"] = -99.
r_mask = (cosmos2020["FLUX_RADIUS"] > 0)
cosmos2020.loc[r_mask, "SIZE"] = 4*cosmos2020.loc[r_mask, "FLUX_RADIUS"]*0.15

cosmos2020["STAR"] = 0

cosmos2020["MOONRISE_ID"] = cosmos2020["COSMOS2020_ID"].astype(str)
cosmos2020["MOONRISE_ID"] = "11001" + cosmos2020["MOONRISE_ID"].str.zfill(9)

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
cosmos2020 = pd.merge(cosmos2020, pipes_cat, how="outer",
                      left_on="COSMOS2020_ID", right_on="ID")

# ##### Merge in Khostovan et al. (2025) v1.1 specz compilation #####

# Khostovan et al. (2025) v1.1 specz compilation, unedited download from
# https://github.com/cosmosastro/speczcompilation
specz_path = "specz_compilation_COSMOS_DR1.1_unique.fits"
khost = Table.read(specz_path).to_pandas()
khost.rename(columns={"specz": "ZSPEC_KHOSTOVAN", "flag": "ZFLAG_KHOSTOVAN"},
             inplace=True)

# Cut Khostovan catalogue to objects that have counterparts in cosmos2020
# catalogue, as well as high confidence speczs (flags 3 and 4)
khost = khost[khost["Id_COS20_Classic"] > 0]
#khost = khost[khost["Confidence_level"] >= 95]

khost = khost[["Id_COS20_Classic", "ZSPEC_KHOSTOVAN", "ZFLAG_KHOSTOVAN"]]

# Match using cosmos2020 and cosmos2025 IDs provided by Khostovan et al.
cosmos2020 = pd.merge(cosmos2020, khost, how="outer", left_on="COSMOS2020_ID",
                      right_on="Id_COS20_Classic")

# ##### Merge in Dawn JWST archive v4.5 specz compilation #####

# DJA v4.5 NIRSpec catalogue, csv downloaded from
# https://s3.amazonaws.com/msaexp-nirspec/extractions/nirspec_public_v4.5.html
# First loaded into topcat, cut to grade == 3 and internal match performed
# within 0.3" to produce groups of spectra corresponding to the same source
dja_cat = Table.read("dja_nirspec_v4.5_grade3_with_groups.fits").to_pandas()

# Get individual best redshift for objects that have multiple spectra
cols = ["ra", "dec", "zfit", "GroupID"]
aggregated_groups = dja_cat[cols].groupby("GroupID").agg(["mean", "std"])

data_array = np.c_[aggregated_groups[("ra", "mean")].values,
                   aggregated_groups[("dec", "mean")].values,
                   aggregated_groups[("zfit", "mean")].values,
                   aggregated_groups[("zfit", "std")].values]

resolved_groups = pd.DataFrame(data_array,
                               columns=["ra", "dec", "zfit", "zfit_std"])

# Only retain objects with multiple spectra where all the redshifts agree
resolved_groups = resolved_groups[resolved_groups["zfit_std"] < 0.05]
resolved_groups = resolved_groups[["ra", "dec", "zfit"]]

# Select rows that were not part of a group
dja_cat = dja_cat[dja_cat["GroupID"].isnull()]
dja_cat = dja_cat[["ra", "dec", "zfit"]]

# Merge groups that have been reduced to a single redshift back into cat
dja_cat = pd.concat([dja_cat, resolved_groups], ignore_index=True, axis=0)


dja_cat.rename(columns={"ra": "ra_dja", "dec": "dec_dja", "zfit": "ZSPEC_DJA"},
             inplace=True)

cosmos2020 = pair_match_sky(cosmos2020, dja_cat, 0.3,
                                  match_selection="Best match, symmetric",
                                  ra_col_1="RA", dec_col_1="DEC",
                                  ra_col_2="ra_dja", dec_col_2="dec_dja",
                                  join_type="All from 1",
                                  suffix1="", suffix2="_dja")

cosmos2020.drop(columns=["ra_dja", "dec_dja"], inplace=True)

# ##### Merge in GAIA star catalogue and flag potential guide stars #####

# GAIA star catalogue, unedited download using only ra/dec criteria from
# https://gea.esac.esa.int/archive/, SQL query text saved with this file
gaia_table = Table.read("gaia_stars_cosmos.fits").to_pandas()

# Cut GAIA table by RA and DEC to COSMOS2020 catalogue area
ra_mask = (gaia_table["ra"] > 149.05) & (gaia_table["ra"] < 151.07)
dec_mask = (gaia_table["dec"] > 1.39) & (gaia_table["dec"] < 3.08)
gaia_table = gaia_table[ra_mask & dec_mask]
gaia_table["MOONRISE_ID"] = np.arange(1, len(gaia_table)+1).astype(str)
gaia_table["MOONRISE_ID"] = "11006" + gaia_table["MOONRISE_ID"].str.zfill(9)
gaia_table["STAR"] = 1

# ##### Make flag for good potential guide stars #####

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
gaia_table["GOOD_STAR"] = 0
gaia_table.loc[gaia_star_mask, "GOOD_STAR"] = 1

gaia_table["HMAG_FLAG"] = -99.
gaia_table = gaia_table[["MOONRISE_ID", "source_id", "ra", "dec",
                         "pmra", "pmdec", "ruwe",
                         "STAR", "GOOD_STAR", "phot_g_mean_mag",
                         "phot_rp_mean_mag", "HMAG_FLAG"]]

# Merge in H-band magnitudes for GAIA stars from cosmos2020
gaia_table_match = pair_match_sky(gaia_table, cosmos2020, 0.2,
                                  match_selection="Best match, symmetric",
                                  join_type="All from 1",
                                  ra_col_2="RA", dec_col_2="DEC",
                                  suffix1="", suffix2="_c2020")

drop_cols = cosmos2020.columns.tolist()
for i in range(len(drop_cols)):
    if drop_cols[i] in gaia_table.columns:
        drop_cols[i] += "_c2020"

drop_cols.append("match_sep_arcsec")
drop_cols.remove("HMAG")

gaia_table_match["HMAG"] = -99.
mask = gaia_table_match["UVISTA_H_MAG_APER3"] > 0
gaia_table_match.loc[mask, "HMAG"] = gaia_table_match.loc[mask, "UVISTA_H_MAG_APER3"]
gaia_table_match.loc[mask, "HMAG_FLAG"] = 0

gaia_table_match.drop(columns=drop_cols, inplace=True)

# Merge in H-band magnitudes for GAIA stars from 2MASS PSC
twomass = pd.read_csv("2mass_psc_cosmos.csv")
gaia_table_match = pair_match_sky(gaia_table_match, twomass, 0.5,
                                  match_selection="Best match, symmetric",
                                  join_type="All from 1",
                                  suffix1="", suffix2="_2mass")

drop_cols = twomass.columns.tolist()
for i in range(len(drop_cols)):
    if drop_cols[i] in gaia_table.columns:
        drop_cols[i] += "_2mass"

drop_cols.append("match_sep_arcsec")

# Merge in H-band magnitudes from 2mass, converting from Vega to AB
mask = (gaia_table_match["h_m"].notnull()) & (gaia_table_match["HMAG"] < 0)
gaia_table_match.loc[mask, "HMAG"] = gaia_table_match.loc[mask, "h_m"] + 1.38
gaia_table_match.loc[mask, "HMAG_FLAG"] = 1

gaia_table_match.drop(columns=drop_cols, inplace=True)

# For GAIA stars without good H magnitudes, set MAG based on polynomial
# fit to Gaia G-band magnitudes for stars with good H magnitudes
x = gaia_table_match["phot_g_mean_mag"]
bad_H_mask = ((gaia_table_match["HMAG"] < 0)
              | (gaia_table_match["HMAG"] > 21)
              | (gaia_table_match["HMAG"] > 0.75*x + 7)
              | (gaia_table_match["HMAG"] < 0.75*x - 0.5))

gaia_table_match.loc[bad_H_mask, "HMAG"] = 0.75*x + 3.5
gaia_table_match.loc[bad_H_mask, "HMAG_FLAG"] = 2
gaia_table_match.loc[bad_H_mask, "GOOD_STAR"] = 0

# ##### Merge GAIA star catalogue into main catalogue #####

# Get rid of anything in cosmos2020 within 1" of a GAIA star
cosmos2020 = pair_match_sky(cosmos2020, gaia_table_match, 1.0,
                            match_selection="All matches",
                            join_type="1 not 2",
                            ra_col_1="RA", dec_col_1="DEC",
                            suffix1="", suffix2="_gaia")

gaia_table_match.rename(columns={"source_id": "GAIA_STAR_ID",
                                 "ra": "RA", "dec": "DEC",
                                 "pmra": "PMRA", "pmdec": "PMDEC",
                                 "ruwe": "RUWE",
                                 "phot_rp_mean_mag": "GAIA_magR",
                                 "phot_g_mean_mag": "GAIA_magG"},
                                 inplace=True)

# Add in GAIA stars to main catalogue
cosmos2020 = pd.concat([cosmos2020, gaia_table_match], ignore_index=True)

# ##### Merge in high-z and AGN (and other?) source catalogues #####


# ##### Sort out best redshift column #####
cosmos2020["ZBEST"] = -99.
cosmos2020.loc[cosmos2020["ZPHOT"] > 0, "ZBEST"] = cosmos2020["ZPHOT"]

zk_mask = (cosmos2020["ZFLAG_KHOSTOVAN"] >= 3) & (cosmos2020["ZFLAG_KHOSTOVAN"] <= 4)
cosmos2020.loc[zk_mask, "ZBEST"] = cosmos2020.loc[zk_mask, "ZSPEC_KHOSTOVAN"]

zdja_mask = (cosmos2020["ZSPEC_DJA"] > 0)
cosmos2020.loc[zdja_mask, "ZBEST"] = cosmos2020.loc[zdja_mask, "ZSPEC_DJA"]


# ##### Keep only necessary columns, set -99s and save to file #####

cosmos2020.fillna(dict(zip(cosmos2020.columns, [-99] * cosmos2020.shape[0])),
                  inplace=True)
"""
mask = (cosmos2020.columns.str.endswith("_APER3")
        | cosmos2020.columns.str.endswith("_FLUX")
        | cosmos2020.columns.str.endswith("_FLUXERR")
        | cosmos2020.columns.str.contains("FLUX"))

flux_cols = cosmos2020.columns[mask].tolist()
"""
cosmos2020 = cosmos2020[["MOONRISE_ID", "COSMOS2020_ID", "GAIA_STAR_ID",
                         "RA", "DEC", "PMRA", "PMDEC", "HMAG", "HMAG_FLAG", "SIZE",
                         "FLAG_COMBINED", "ZBEST", "ZPHOT", "ZSPEC_KHOSTOVAN",
                         "ZFLAG_KHOSTOVAN", "ZSPEC_DJA", "STAR", "GOOD_STAR", "RUWE",
                         "GAIA_magG", "GAIA_magR", "ABS_U", "ABS_V", "ABS_J",
                         "stellar_mass_16", "stellar_mass_50",
                         "stellar_mass_84"]]# + flux_cols]

cosmos2020.sort_values("MOONRISE_ID", inplace=True)

Table.from_pandas(cosmos2020).write("moonrise_cosmos_catalogue.fits",
                                    overwrite=True)
