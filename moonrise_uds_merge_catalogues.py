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

# derek uds catalogue from 2021 paper
derek_uds = Table.read("XMM1+2+3_Hband_3sigma_z0p0_z4p0_101122.fits").to_pandas()

derek_uds["PMRA"] = 0.
derek_uds["PMDEC"] = 0.
derek_uds.rename(columns={"zphot": "ZPHOT", "Dec": "DEC",
                           "ID": "derek_uds_ID"}, inplace=True)

derek_uds["HMAG"] = 0.
uds_mask = (derek_uds["derek_uds_ID"] > 1e6) & (derek_uds["derek_uds_ID"] < 2e6)
derek_uds.loc[uds_mask, "HMAG"] = 23.9-2.5*np.log10(derek_uds["wfcam_H"]*derek_uds["isofactor"]*derek_uds["iso_to_total"])
derek_uds.loc[np.invert(uds_mask), "HMAG"] = 23.9-2.5*np.log10(derek_uds["VIDEO_H"]*derek_uds["isofactor"]*derek_uds["iso_to_total"])

derek_uds["HMAG_FLAG"] = 0

derek_uds["SIZE"] = -99.

derek_uds["STAR"] = 0

derek_uds["MOONRISE_ID"] = derek_uds["derek_uds_ID"].astype(str)
derek_uds["MOONRISE_ID"] = "13001" + derek_uds["MOONRISE_ID"].str.zfill(9)

# ##### Merge in bagpipes results catalogue #####

# Bagpipes fit results for derek_uds sub-sample
pipes_cat = Table.read("xmmlss_bagpipes_subsample.fits").to_pandas()

pipes_cat = pipes_cat[["ID", "stellar_mass_16", "stellar_mass_50",
                       "stellar_mass_84", "U_50", "V_50", "J_50"]]

pipes_cat.rename(columns={"U_50": "ABS_U", "V_50": "ABS_V",
                          "J_50": "ABS_J"}, inplace=True)

# Merge derek_uds catalogue with Bagpipes fit results catalogue
derek_uds = pd.merge(derek_uds, pipes_cat, how="outer",
                      left_on="derek_uds_ID", right_on="ID")

# ##### Merge in GAIA star catalogue and flag potential acquisition stars #####

# GAIA star catalogue, unedited download using only ra/dec criteria from
# https://gea.esac.esa.int/archive/, SQL query text saved with this file
gaia_table = Table.read("gaia_stars_uds.fits").to_pandas()

# Cut GAIA table by DEC to derek uds catalogue area
dec_mask = (gaia_table["dec"] > -5.7) & (gaia_table["dec"] < -3.9)
gaia_table = gaia_table[dec_mask]
gaia_table["MOONRISE_ID"] = np.arange(1, len(gaia_table)+1).astype(str)
gaia_table["MOONRISE_ID"] = "13006" + gaia_table["MOONRISE_ID"].str.zfill(9)
gaia_table["STAR"] = 1

# ##### Make flag for good potential guide stars #####

# Cut by ruwe value to exclude binaries using gaiaunlimited package to
# calculate threshold local to field from Castro-Ginard et al. (2024)
# https://doi.org/10.1051/0004-6361/202450172
sf = binaries.BinarySystemsSelectionFunction()

# Central  coordinates for the COSMOS 2020 catalogue
central_coord = SkyCoord(ra=35.5*u.degree, dec=-4.9*u.degree, frame="icrs")

# Returns the RUWE threshold above which a source is considered a
# potential binary, e.g., see Fig. 3 in Castro-Ginard et al. (2024)
ruwe_threshold = sf.query_RUWE(central_coord, crowding=True)

# Define masks to exclude stars we don't want for various reasons
ruwe_mask = (gaia_table["ruwe"] < ruwe_threshold)
single_mask = (gaia_table["non_single_star"] == 0)
pm_mask = (gaia_table["pmra"].abs() < 0.1*1000)
pm_mask = pm_mask & (gaia_table["pmdec"].abs() < 0.1*1000)
var_flag = (gaia_table["phot_variable_flag"] != "VARIABLE")
mag_mask = (gaia_table["phot_rp_mean_mag"] < 20)# & (gaia_table["phot_rp_mean_mag"] > 15) # Removed in response to request following first commissioning run

# combine masks and apply to GAIA table
gaia_star_mask = ruwe_mask & single_mask & pm_mask & var_flag & mag_mask
gaia_table["GOOD_STAR"] = 0
gaia_table.loc[gaia_star_mask, "GOOD_STAR"] = 1

gaia_table["HMAG_FLAG"] = -99.
gaia_table = gaia_table[["MOONRISE_ID", "source_id", "ra", "dec",
                         "pmra", "pmdec", "ruwe",
                         "STAR", "GOOD_STAR", "phot_g_mean_mag",
                         "phot_rp_mean_mag", "HMAG_FLAG"]]

# Merge in H-band magnitudes for GAIA stars from derek_uds
gaia_table_match = pair_match_sky(gaia_table, derek_uds, 0.2,
                                  match_selection="Best match, symmetric",
                                  join_type="All from 1",
                                  ra_col_2="RA", dec_col_2="DEC",
                                  suffix1="", suffix2="_derek_uds")

drop_cols = derek_uds.columns.tolist()
for i in range(len(drop_cols)):
    if drop_cols[i] in gaia_table.columns:
        drop_cols[i] += "_derek_uds"

drop_cols.append("match_sep_arcsec")
drop_cols.remove("HMAG")

mask = gaia_table_match["HMAG"] > 0
gaia_table_match.loc[mask, "HMAG_FLAG"] = 0

gaia_table_match.drop(columns=drop_cols, inplace=True)

# Merge in H-band magnitudes for GAIA stars from 2MASS PSC
twomass = pd.read_csv("2mass_psc_uds.csv")
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
# The AB to Vega conversion of 1.27 was determined empirically by comparing
# 2mass and Derek's UDS H-band magnitudes for objects in both catalogues
mask = (gaia_table_match["h_m"].notnull()) & ((gaia_table_match["HMAG"] < 0) | (gaia_table_match["HMAG"].isnull()))

gaia_table_match.loc[mask, "HMAG"] = gaia_table_match.loc[mask, "h_m"] + 1.27
gaia_table_match.loc[mask, "HMAG_FLAG"] = 1

gaia_table_match.drop(columns=drop_cols, inplace=True)

# For GAIA stars without good H magnitudes, set MAG based on polynomial
# fit to Gaia G-band magnitudes for stars with good H magnitudes
x = gaia_table_match["phot_g_mean_mag"]
bad_H_mask = ((gaia_table_match["HMAG"] < 0)
              | (gaia_table_match["HMAG"].isnull())
              | (gaia_table_match["HMAG"] > 21)
              | (gaia_table_match["HMAG"] > 0.75*x + 7)
              | (gaia_table_match["HMAG"] < 0.75*x - 0.5))

gaia_table_match.loc[bad_H_mask, "HMAG"] = 0.75*x + 3.42
gaia_table_match.loc[bad_H_mask, "HMAG_FLAG"] = 2
gaia_table_match.loc[bad_H_mask, "GOOD_STAR"] = 0

# For anything that still doesn't have an H magnitude, set HMAG = 98
still_bad_H_mask = gaia_table_match["HMAG"].isnull() | (gaia_table_match["HMAG"] < 0)

gaia_table_match.loc[still_bad_H_mask, "HMAG"] = 98.
gaia_table_match.loc[still_bad_H_mask, "HMAG_FLAG"] = -99

# ##### Merge GAIA star catalogue into main catalogue #####

# Get rid of anything in derek_uds within 1" of a GAIA star
derek_uds = pair_match_sky(derek_uds, gaia_table_match, 1.0,
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
derek_uds = pd.concat([derek_uds, gaia_table_match], ignore_index=True)

# ##### Merge in Derek i-band catalogue sources
derek_iband = Table.read("XMM1+2+3_full_i_band_plus_H_161222.fits").to_pandas()
nohmask = (derek_iband["ID_H_band"] == -99)
derek_iband = derek_iband[nohmask]
derek_iband["MOONRISE_ID"] = "13002" + derek_iband["ID_i_band"].astype(str).str.zfill(9)
derek_iband.rename(columns={"RA_i_band": "RA", "Dec_i_band": "DEC"}, inplace=True)
derek_iband["HMAG"] = 98
derek_iband["SIZE"] = -99

derek_uds = pd.concat([derek_uds, derek_iband], ignore_index=True)
print(derek_uds.shape)

# ##### Merge in high-z and AGN (and other?) source catalogues #####

agn_cat = Table.read("XMMLSS_WG5_AGN_commissioning_v2.fits").to_pandas()
agn_cat.rename(columns={"ZSPEC_OTHER": "ZSPEC_AGN", "TEXP": "AGN_TEXP", "ID_OTHER_REF": "AGN_ID_OTHER_REF", "FLAG_MAG": "AGN_FLAG_MAG"}, inplace=True)


# There are probably other columns from the AGN catalogue we should keep,
# needs discussing with AGN working group
#agn_cat = agn_cat[["RA", "DEC", "AGN_TEXP", "AGN_ID_OTHER_REF", "ZSPEC_AGN"]]

# Get rid of anything in the Derek UDS catalogue that's also in the AGN catalogue
derek_uds = pair_match_sky(derek_uds, agn_cat, 0.3,
                           match_selection="All matches",
                           join_type="1 not 2",
                           ra_col_1="RA", dec_col_1="DEC",
                           ra_col_2="RA", dec_col_2="DEC",
                           suffix1="", suffix2="_agn_catalogue")

agn_cat.reset_index(drop=True, inplace=True)
agn_cat.index += 1
agn_cat["MOONRISE_ID"] = "13004" + agn_cat.index.astype(str).str.zfill(9)
agn_cat["HMAG"] = agn_cat["Hmag_FINAL"] + 2.5*np.log10(1/0.9) # AGN cat contains H_auto, we're taking H_total to be H_auto/0.9
agn_cat["SIZE"] = -99

derek_uds = pd.concat([derek_uds, agn_cat], ignore_index=True)

# Now Hiz
hiz_cat = Table.read("LAES_UDS_moonsinfo.fits").to_pandas()
hiz_cat["LAE_ID"] = hiz_cat["LAE_ID"].astype(str)
hiz_cat.rename(columns={"ZSPEC": "ZSPEC_HIZ", "LAE_ID": "HIZ_ID"}, inplace=True)
hiz_cat = hiz_cat[["HIZ_ID", "RA", "DEC", "ZSPEC_HIZ"]]

# Select things in both the High-z catalogue and Derek's catalogue
hiz_cat_old_obj = pair_match_sky(hiz_cat, derek_uds, 0.3,
                                 match_selection="Best match, symmetric",
                                 join_type="1 and 2",
                                 ra_col_1="RA", dec_col_1="DEC",
                                 ra_col_2="RA", dec_col_2="DEC",
                                 suffix1="", suffix2="_derek_uds")

hiz_cat_old_obj.index = hiz_cat_old_obj["MOONRISE_ID"].values
hiz_cat_old_obj = hiz_cat_old_obj[["MOONRISE_ID", "HIZ_ID", "ZSPEC_HIZ"]]

# Merge hiz columns into Derek catalogue for objects in both
derek_uds.index = derek_uds["MOONRISE_ID"].values
derek_uds = pd.merge(derek_uds, hiz_cat_old_obj, how="left", left_index=True, right_index=True, suffixes=(None, "_hiz"))
derek_uds.loc[derek_uds["HIZ_ID"].isnull(), "HIZ_ID"] = "-99"

# Get rid of anything in the Hiz catalogue that's already in derek uds cat
hiz_cat_new_obj = pair_match_sky(hiz_cat, derek_uds, 0.3,
                                 match_selection="Best match, symmetric",
                                 join_type="1 not 2",
                                 ra_col_1="RA", dec_col_1="DEC",
                                 ra_col_2="RA", dec_col_2="DEC",
                                 suffix1="", suffix2="_derek_uds")

# Add these things to the catalogue by concatenation
hiz_cat_new_obj.reset_index(drop=True, inplace=True)
hiz_cat_new_obj.index += 1
hiz_cat_new_obj["MOONRISE_ID"] = "13003" + hiz_cat_new_obj.index.astype(str).str.zfill(9)
hiz_cat_new_obj["HMAG"] = 98
hiz_cat_new_obj["SIZE"] = -99

derek_uds = pd.concat([derek_uds, hiz_cat_new_obj], ignore_index=True)

# ##### Merge in spectroscopic redshift catalogues #####

# ##### Merge in UDSz catalogue #####
udsz_cat = Table.read("UDSz-secure-March2014.fits").to_pandas()
udsz_cat.rename(columns={"DR1ID": "UDSz_ID", "RA": "ra_udsz", "DEC": "dec_udsz",
                         "redshift": "ZSPEC_UDSZ", "Flag": "ZFLAG_UDSZ"}, inplace=True)

udsz_cat = udsz_cat[["UDSz_ID", "ra_udsz", "dec_udsz", "ZSPEC_UDSZ", "ZFLAG_UDSZ"]]

derek_uds = pair_match_sky(derek_uds, udsz_cat, 0.3,
                                  match_selection="Best match, symmetric",
                                  ra_col_1="RA", dec_col_1="DEC",
                                  ra_col_2="ra_udsz", dec_col_2="dec_udsz",
                                  join_type="All from 1",
                                  suffix1="", suffix2="_udsz")

derek_uds.drop(columns=["ra_udsz", "dec_udsz"], inplace=True)

# #### Merge in Galametz et al. (2018) catalogue #####

galametz_cols = ["GALAMETZ18_ID", "RA_hh", "RA_mm", "RA_ss", "DEC_dd", "DEC_mm", "DEC_ss", "ZSPEC_GALAMETZ18", "ZFLAG_GALAMETZ18", "type"]
galametz_cat = pd.read_csv("galametz18_table2.txt", names=galametz_cols, sep="\s+")
galametz_cat["ra_galametz"] = (galametz_cat["RA_hh"]*15 + galametz_cat["RA_mm"]/4 + galametz_cat["RA_ss"]/240)
galametz_cat["dec_galametz"] = (galametz_cat["DEC_dd"] + galametz_cat["DEC_mm"]/60 + galametz_cat["DEC_ss"]/3600)

galametz_cat = galametz_cat[["GALAMETZ18_ID", "ra_galametz", "dec_galametz", "ZSPEC_GALAMETZ18", "ZFLAG_GALAMETZ18"]]

derek_uds = pair_match_sky(derek_uds, galametz_cat, 0.3,
                                  match_selection="Best match, symmetric",
                                  ra_col_1="RA", dec_col_1="DEC",
                                  ra_col_2="ra_galametz", dec_col_2="dec_galametz",
                                  join_type="All from 1",
                                  suffix1="", suffix2="_galametz")

derek_uds.drop(columns=["ra_galametz", "dec_galametz"], inplace=True)

# ##### Merge in VANDELS catalogue #####

vandels_cat = Table.read("vandels_dr4_cat.fits").to_pandas()
zflag_mask = ((vandels_cat["zflg"] == 3) | (vandels_cat["zflg"] == 4)
              | (vandels_cat["zflg"] == 9) | (vandels_cat["zflg"] == 19)
              | (vandels_cat["zflg"] == 13) | (vandels_cat["zflg"] == 14)
              | (vandels_cat["zflg"] == 23) | (vandels_cat["zflg"] == 24)
              | (vandels_cat["zflg"] == 214) | (vandels_cat["zflg"] == 29))

vandels_cat = vandels_cat[zflag_mask]
vandels_cat.rename(columns={"id": "VANDELS_ID", "alpha": "ra_vandels",
                            "delta": "dec_vandels", "zspec": "ZSPEC_VANDELS",
                            "zflg": "ZFLAG_VANDELS"}, inplace=True)

vandels_cat = vandels_cat[["VANDELS_ID", "ra_vandels", "dec_vandels",
                           "ZSPEC_VANDELS", "ZFLAG_VANDELS"]]

derek_uds = pair_match_sky(derek_uds, vandels_cat, 0.3,
                                  match_selection="Best match, symmetric",
                                  ra_col_1="RA", dec_col_1="DEC",
                                  ra_col_2="ra_vandels", dec_col_2="dec_vandels",
                                  join_type="All from 1",
                                  suffix1="", suffix2="_vandels")

derek_uds.drop(columns=["ra_vandels", "dec_vandels"], inplace=True)

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
# This only removed ~0.3 per cent of objects (groups) from the DJA catalogue
resolved_groups = resolved_groups[resolved_groups["zfit_std"] < 0.05]
resolved_groups = resolved_groups[["ra", "dec", "zfit"]]

# Select rows that were not part of a group
dja_cat = dja_cat[dja_cat["GroupID"].isnull()]
dja_cat["progid"] = dja_cat["msamet"].str[2:7].astype(int)
dja_cat = dja_cat[["ra", "dec", "zfit", "srcid", "progid"]]

# Merge groups that have been reduced to a single redshift back into cat
dja_cat = pd.concat([dja_cat, resolved_groups], ignore_index=True, axis=0)

dja_cat.rename(columns={"ra": "ra_dja", "dec": "dec_dja", "zfit": "ZSPEC_DJA",
                        "srcid": "DJA_SOURCE_ID", "progid": "DJA_PROG_ID"},
               inplace=True)

derek_uds = pair_match_sky(derek_uds, dja_cat, 0.3,
                                  match_selection="Best match, symmetric",
                                  ra_col_1="RA", dec_col_1="DEC",
                                  ra_col_2="ra_dja", dec_col_2="dec_dja",
                                  join_type="All from 1",
                                  suffix1="", suffix2="_dja")

derek_uds.drop(columns=["ra_dja", "dec_dja"], inplace=True)

# ##### Sort out best redshift column #####
derek_uds["ZBEST"] = -99.
derek_uds["ZBEST_FLAG"] = -99.

derek_uds.loc[derek_uds["ZPHOT"] > 0, "ZBEST"] = derek_uds["ZPHOT"]
derek_uds.loc[derek_uds["ZPHOT"] > 0, "ZBEST_FLAG"] = 0

zudsz_mask = (derek_uds["ZSPEC_UDSZ"] > 0)
derek_uds.loc[zudsz_mask, "ZBEST"] = derek_uds.loc[zudsz_mask, "ZSPEC_UDSZ"]
derek_uds.loc[zudsz_mask, "ZBEST_FLAG"] = 1

zgalametz_mask = (derek_uds["ZSPEC_GALAMETZ18"] > 0)
derek_uds.loc[zgalametz_mask, "ZBEST"] = derek_uds.loc[zgalametz_mask, "ZSPEC_GALAMETZ18"]
derek_uds.loc[zgalametz_mask, "ZBEST_FLAG"] = 2

zvandels_mask = (derek_uds["ZSPEC_VANDELS"] > 0)
derek_uds.loc[zvandels_mask, "ZBEST"] = derek_uds.loc[zvandels_mask, "ZSPEC_VANDELS"]
derek_uds.loc[zvandels_mask, "ZBEST_FLAG"] = 3

zdja_mask = (derek_uds["ZSPEC_DJA"] > 0)
derek_uds.loc[zdja_mask, "ZBEST"] = derek_uds.loc[zdja_mask, "ZSPEC_DJA"]
derek_uds.loc[zdja_mask, "ZBEST_FLAG"] = 4

zhiz_mask = (derek_uds["ZSPEC_HIZ"] > 0)
derek_uds.loc[zhiz_mask, "ZBEST"] = derek_uds.loc[zhiz_mask, "ZSPEC_HIZ"]
derek_uds.loc[zhiz_mask, "ZBEST_FLAG"] = 5

zagn_mask = (derek_uds["ZSPEC_AGN"] > 0)
derek_uds.loc[zagn_mask, "ZBEST"] = derek_uds.loc[zagn_mask, "ZSPEC_AGN"]
derek_uds.loc[zagn_mask, "ZBEST_FLAG"] = 6

# ##### Add milky way dust

from dustmaps.sfd import SFDQuery
from astropy.coordinates import SkyCoord

def get_mw_ebv(ra,dec):
    coords = SkyCoord(ra, dec, unit='deg')
    sfd = SFDQuery() #Schlegel, Finkbeiner & Davis (1998)
    ebv = sfd(coords)
    return ebv * 0.86 # Apply recalibration of Schlafly & Finkbeiner (2011)

#Get ra and dec from somewhere, then run function

derek_uds["MW_EBV"] = get_mw_ebv(derek_uds["RA"], derek_uds["DEC"])

# ##### Final tidying up and writing output catalogue #####

derek_uds = derek_uds[["MOONRISE_ID", "derek_uds_ID", "GAIA_STAR_ID", "UDSz_ID",
                       "GALAMETZ18_ID", "VANDELS_ID", "HIZ_ID", "DJA_SOURCE_ID", "DJA_PROG_ID",
                         "RA", "DEC", "PMRA", "PMDEC", "HMAG", "HMAG_FLAG", "SIZE",
                         "ZBEST", "ZBEST_FLAG", "ZPHOT", "ZSPEC_UDSZ", "ZFLAG_UDSZ",
                         "ZSPEC_GALAMETZ18", "ZFLAG_GALAMETZ18",
                         "ZSPEC_VANDELS", "ZFLAG_VANDELS", "ZSPEC_DJA",
                         "STAR", "GOOD_STAR", "RUWE",
                         "GAIA_magG", "GAIA_magR", "ABS_U", "ABS_V", "ABS_J",
                         "stellar_mass_16", "stellar_mass_50",
                         "stellar_mass_84", "AGN_TEXP", "ZSPEC_AGN","AGN_ID_OTHER_REF", "AGN_FLAG_MAG", "MW_EBV"]]

for col in ["derek_uds_ID", "GAIA_STAR_ID", "HMAG_FLAG"]:
    nan_mask = derek_uds[col].isnull()
    derek_uds.loc[nan_mask, col] = -99

for col in ["STAR", "GOOD_STAR"]:
    nan_mask = derek_uds[col].isnull()
    derek_uds.loc[nan_mask, col] = 0

derek_uds.sort_values("MOONRISE_ID", inplace=True)
derek_uds["MOONRISE_ID"] = derek_uds["MOONRISE_ID"].astype(int)

for i in range(len(derek_uds.columns)):
    col = derek_uds.columns[i]
    dtype = derek_uds.dtypes.values[i]
    nan_mask = derek_uds[col].isnull()
    if dtype == "float64" or dtype == "int64" or dtype == "float32" or dtype == "int32":
        derek_uds.loc[nan_mask, col] = -99
    elif dtype == "str" or dtype == "object":
        derek_uds.loc[nan_mask, col] = "-99"


for id_col in ["MOONRISE_ID", "derek_uds_ID", "GAIA_STAR_ID", "HIZ_ID", "HMAG_FLAG",
               "ZBEST_FLAG", "STAR", "GOOD_STAR"]:
    derek_uds[id_col] = derek_uds[id_col].astype(int)

Table.from_pandas(derek_uds).write("moonrise_uds_xmm_catalogue.fits",
                                    overwrite=True)
