import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.table import Table
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy import units as u


def positional_cross_match(cat1, cat2, dist_arcsec, how="inner", ra_col_1="ra",
                           dec_col_1="dec", ra_col_2="ra", dec_col_2="dec",
                           suffix1="_1", suffix2="_2"):
    """
    Cross-match two catalogues based on ra and dec columns, returning
    a new joined catalogue with only sources that have a close match.
    Reproduces functionality of TOPCAT tool's sky match algorithm with
    match seclection: "best match, symmetric" and join type: "1 and 2"
    """

    coords1 = SkyCoord(ra=cat1[ra_col_1].values*u.degree,
                       dec=cat1[dec_col_1].values*u.degree)
    coords2 = SkyCoord(ra=cat2[ra_col_2].values*u.degree,
                       dec=cat2[dec_col_2].values*u.degree)

    result = match_coordinates_sky(coords1, coords2)

    dist_mask = result[1] < dist_arcsec*u.arcsec

    matched = pd.merge(cat1[dist_mask].reset_index(drop=True),
                       cat2.iloc[result[0][dist_mask]].reset_index(drop=True),
                       left_index=True, right_index=True, how=how,
                       suffixes=(suffix1, suffix2))

    matched["cat2_row"] = result[0][dist_mask]
    matched["match_sep_arcsec"] = result[1][dist_mask]

    idxmin = matched.groupby("cat2_row")["match_sep_arcsec"].idxmin()
    matched = matched.loc[idxmin].reset_index(drop=True)

    return matched


# ##### Load up base tables #####

# COSMOS2020 catalogue, unedited download from
# https://irsa.ipac.caltech.edu/data/COSMOS/tables/cosmos2020/
c2020_path = "../COSMOS2020_CLASSIC_R1_v2.2_p3.fits"
cosmos2020_raw = Table.read(c2020_path).to_pandas()

cosmos2020_raw.rename(columns={"ALPHA_J2000": "ra",
                               "DELTA_J2000": "dec"}, inplace=True)

# Cut by photometry flag and H-band magnitude to match MOONRISE criteria
mask_phot_combined = (cosmos2020_raw["FLAG_COMBINED"] == 0)
mask_h24 = (cosmos2020_raw["UVISTA_H_MAG_APER3"] <= 24)
both_mask = mask_h24 & mask_phot_combined
cosmos2020 = cosmos2020_raw[both_mask].reset_index(drop=True)

# COSMOS2025 photometric catalogue, unedited download from
# https://cosmos2025.iap.fr/catalog_download.php
c2025_flux_path = "../COSMOSWeb_mastercatalog_v1.1_photom_primary.fits"
cosmos2025_fluxes = Table.read(c2025_flux_path)

# Fix weird column formatting
aperture_diameters = [0.2, 0.3, 0.5, 0.75]
aper_cols = [col for col in cosmos2025_fluxes.colnames if "_aper_" in col]

for aper_col in aper_cols:
    for i in range(len(aperture_diameters)):
        diam = aperture_diameters[i]
        new_col_name = aper_col + f"_{diam}_arcsec"
        cosmos2025_fluxes[new_col_name] = cosmos2025_fluxes[aper_col][:, i]

    del cosmos2025_fluxes[aper_col]

cosmos2025_fluxes = cosmos2025_fluxes.to_pandas()

# COSMOS2025 photoz catalogue, unedited download from
# https://cosmos2025.iap.fr/catalog_download.php
c2025_photoz_path = "../COSMOSWeb_mastercatalog_v1.1_lephare.fits"
cosmos2025_photoz = Table.read(c2025_photoz_path).to_pandas()

cosmos2025_raw = pd.merge(cosmos2025_fluxes, cosmos2025_photoz,
                          left_index=True, right_index=True)

# Cut both catalogues to a common set of objects
cosmos2020 = positional_cross_match(cosmos2020, cosmos2025_raw, 0.5,
                                    suffix1="", suffix2="_cosmos2025")

cosmos2025 = positional_cross_match(cosmos2025_raw, cosmos2020, 0.5,
                                    suffix1="", suffix2="_cosmos2020")

# DJA v4.5 NIRSpec catalogue, unedited csv download from
# https://s3.amazonaws.com/msaexp-nirspec/extractions/nirspec_public_v4.5.html
dja_cat = pd.read_csv("../dja_nirspec_v4.5.csv")
dja_cat = dja_cat[dja_cat["grade"] == 3]

# ##### Compare photozs to DJA v4.5 specz compilation #####

cosmos2020_dja = positional_cross_match(cosmos2020, dja_cat, 0.3)
cosmos2025_dja = positional_cross_match(cosmos2025, dja_cat, 0.3,
                                        ra_col_1="ra_cosmos2020",
                                        dec_col_1="dec_cosmos2020")

moonrise_zphot_mask_cosmos2020 = ((cosmos2020_dja["ez_z_phot"] >= 0.7)
                                  & (cosmos2020_dja["ez_z_phot"] <= 2.6))

moonrise_zphot_mask_cosmos2025 = ((cosmos2025_dja["zfinal"] >= 0.7)
                                  & (cosmos2025_dja["zfinal"] <= 2.6))

moonrise_zdja_mask_cosmos2020 = ((cosmos2020_dja["zfit"] >= 0.7)
                                 & (cosmos2020_dja["zfit"] <= 2.6))

moonrise_zdja_mask_cosmos2025 = ((cosmos2025_dja["zfit"] >= 0.7)
                                 & (cosmos2025_dja["zfit"] <= 2.6))

# Make plot
fig = plt.figure(figsize=(12, 5))
gs = fig.add_gridspec(1, 2)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])

ax1.set_title("COSMOS2020 vs DJA v4.5")
ax2.set_title("COSMOS2025 vs DJA v4.5")

ax1.scatter(cosmos2020_dja["ez_z_phot"], cosmos2020_dja["zfit"], s=10,
            alpha=0.5)
ax2.scatter(cosmos2025_dja["zfinal"], cosmos2025_dja["zfit"], s=10,
            alpha=0.5)

axesmax = np.max([np.max(cosmos2020_dja["ez_z_phot"]),
                  np.max(cosmos2020_dja["zfit"]),
                  np.max(cosmos2025_dja["zfinal"]),
                  np.max(cosmos2025_dja["zfit"])])

ax1.set_xlabel("COSMOS2020 ez_z_phot")
ax1.set_ylabel("DJA v4.5 zfit")

ax2.set_xlabel("COSMOS2025 zfinal")
ax2.set_ylabel("DJA v4.5 zfit")

ax1.set_xlim(0, axesmax+0.5)
ax2.set_xlim(0, axesmax+0.5)
ax1.set_ylim(0, axesmax+0.5)
ax2.set_ylim(0, axesmax+0.5)

ax1.annotate(f"N={len(cosmos2020_dja)}", xy=(0.05, 0.95),
             xycoords="axes fraction", fontsize=14, ha="left", va="top")

ax2.annotate(f"N={len(cosmos2025_dja)}", xy=(0.05, 0.95),
             xycoords="axes fraction", fontsize=14, ha="left", va="top")

mask1 = moonrise_zphot_mask_cosmos2020 & ~moonrise_zdja_mask_cosmos2020
ax1.annotate("$z_\\mathrm{phot}$" + f" wrongly within 0.7$-$2.6 range: "
             + f"{100*np.sum(mask1)/len(cosmos2020_dja):.1f} \\%",
             xy=(0.05, 0.9), xycoords="axes fraction", fontsize=14,
             ha="left", va="top")

mask2 = moonrise_zphot_mask_cosmos2025 & ~moonrise_zdja_mask_cosmos2025
ax2.annotate("$z_\\mathrm{phot}$ wrongly within 0.7$-$2.6 range: "
             + f"{100*np.sum(mask2)/len(cosmos2025_dja):.1f} \\%",
             xy=(0.05, 0.9), xycoords="axes fraction", fontsize=14,
             ha="left", va="top")

mask3 = ~moonrise_zphot_mask_cosmos2020 & moonrise_zdja_mask_cosmos2020
ax1.annotate("$z_\\mathrm{phot}$ wrongly outside 0.7$-$2.6 range: "
             + f"{100*np.sum(mask3)/len(cosmos2020_dja):.1f} \\%",
             xy=(0.05, 0.85), xycoords="axes fraction", fontsize=14,
             ha="left", va="top")

mask4 = ~moonrise_zphot_mask_cosmos2025 & moonrise_zdja_mask_cosmos2025
ax2.annotate("$z_\\mathrm{phot}$ wrongly outside 0.7$-$2.6 range: "
             + f"{100*np.sum(mask4)/len(cosmos2025_dja):.1f} \\%",
             xy=(0.05, 0.85), xycoords="axes fraction", fontsize=14,
             ha="left", va="top")

plt.savefig("cosmos2020_2025_photozs_vs_dja_v4.5_speczs.pdf", dpi=300,
            bbox_inches="tight")

# ##### Now compare photozs to Khostovan et al. (2025) specz compilation #####

# Khostovan et al. (2025) v1.1 specz compilation, unedited download from
# https://github.com/cosmosastro/speczcompilation
specz_path = "../specz_compilation_COSMOS_DR1.1_unique.fits"
khost = Table.read(specz_path).to_pandas()
khost.rename(columns={"specz": "specz_khost", "ra_corrected": "ra",
                      "dec_corrected": "dec"}, inplace=True)

# Cut Khostovan catalogue to objects that have counterparts in both cosmos2020
# and cosmos2025 catalogues, as well as high confidence speczs (flags 3 and 4)
both_cats_mask = (khost["Id_COS20_Classic"] > 0) & (khost["Id_COSMOS25"] > 0)
khost = khost[both_cats_mask]

khost = khost[khost["Confidence_level"] >= 95]

# Match using cosmos2020 and cosmos2025 IDs provided by Khostovan et al.
cosmos2020_khost = pd.merge(khost, cosmos2020_raw, left_on="Id_COS20_Classic",
                            right_on="ID", how="inner", suffixes=("_1", "_2"))

cosmos2025_khost = pd.merge(khost, cosmos2025_raw, left_on="Id_COSMOS25",
                            right_on="id", how="inner", suffixes=("_1", "_2"))

# Cut by photometry flag and H-band magnitude to match MOONRISE criteria
mask_phot_combined = (cosmos2020_khost["FLAG_COMBINED"] == 0)
mask_h24 = (cosmos2020_khost["UVISTA_H_MAG_APER3"] <= 24)
both_mask = mask_h24 & mask_phot_combined
cosmos2020_khost = cosmos2020_khost[both_mask]
cosmos2025_khost = cosmos2025_khost[both_mask]

"""
# Alternatively could do our own matching, Khostovan uses a 1.5" radius
# for COSMOS2020 but must use a smaller one for COSMOS2025 though exact
# value not clear - I've tested different radii, makes little difference
cosmos2020_khost = positional_cross_match(cosmos2020, khost, 0.5)
cosmos2025_khost = positional_cross_match(cosmos2025, khost, 0.5)
"""

moonrise_zphot_mask_cosmos2020 = ((cosmos2020_khost["ez_z_phot"] >= 0.7)
                                  & (cosmos2020_khost["ez_z_phot"] <= 2.6))

moonrise_zphot_mask_cosmos2025 = ((cosmos2025_khost["zfinal"] >= 0.7)
                                  & (cosmos2025_khost["zfinal"] <= 2.6))

moonrise_zkhost_mask_cosmos2020 = ((cosmos2020_khost["specz_khost"] >= 0.7)
                                   & (cosmos2020_khost["specz_khost"] <= 2.6))

moonrise_zkhost_mask_cosmos2025 = ((cosmos2025_khost["specz_khost"] >= 0.7)
                                   & (cosmos2025_khost["specz_khost"] <= 2.6))

# Make plot
fig = plt.figure(figsize=(12, 5))
gs = fig.add_gridspec(1, 2)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])

ax1.set_title("COSMOS2020 vs Khostovan et al. (2025)")
ax2.set_title("COSMOS2025 vs Khostovan et al. (2025)")

ax1.scatter(cosmos2020_khost["ez_z_phot"],
            cosmos2020_khost["specz_khost"], s=5, alpha=0.2)
ax2.scatter(cosmos2025_khost["zfinal"],
            cosmos2025_khost["specz_khost"], s=5, alpha=0.2)

axesmax = np.max([np.max(cosmos2020_khost["ez_z_phot"]),
                  np.max(cosmos2020_khost["specz_khost"]),
                  np.max(cosmos2025_khost["zfinal"]),
                  np.max(cosmos2025_khost["specz_khost"])])

ax1.set_xlabel("COSMOS2020 ez_z_phot")
ax1.set_ylabel("Khostovan et al. (2025) specz")

ax2.set_xlabel("COSMOS2025 zfinal")
ax2.set_ylabel("Khostovan et al. (2025) specz")

ax1.set_xlim(0, axesmax+0.5)
ax2.set_xlim(0, axesmax+0.5)
ax1.set_ylim(0, axesmax+0.5)
ax2.set_ylim(0, axesmax+0.5)

ax1.annotate(f"N={len(cosmos2020_khost)}", xy=(0.05, 0.95),
             xycoords="axes fraction", fontsize=14, ha="left", va="top")

ax2.annotate(f"N={len(cosmos2025_khost)}", xy=(0.05, 0.95),
             xycoords="axes fraction", fontsize=14, ha="left", va="top")

mask1 = moonrise_zphot_mask_cosmos2020 & ~moonrise_zkhost_mask_cosmos2020
ax1.annotate("$z_\\mathrm{phot}$ wrongly within 0.7$-$2.6 range: "
             + f"{100*np.sum(mask1)/len(cosmos2020_khost):.1f} \\%",
             xy=(0.05, 0.9), xycoords="axes fraction", fontsize=14,
             ha="left", va="top")

mask2 = moonrise_zphot_mask_cosmos2025 & ~moonrise_zkhost_mask_cosmos2025
ax2.annotate("$z_\\mathrm{phot}$ wrongly within 0.7$-$2.6 range: "
             + f"{100*np.sum(mask2)/len(cosmos2025_khost):.1f} \\%",
             xy=(0.05, 0.9), xycoords="axes fraction", fontsize=14,
             ha="left", va="top")

mask3 = ~moonrise_zphot_mask_cosmos2020 & moonrise_zkhost_mask_cosmos2020
ax1.annotate("$z_\\mathrm{phot}$ wrongly outside 0.7$-$2.6 range: "
             + f"{100*np.sum(mask3)/len(cosmos2020_khost):.1f} \\%",
             xy=(0.05, 0.85), xycoords="axes fraction", fontsize=14,
             ha="left",  va="top")

mask4 = ~moonrise_zphot_mask_cosmos2025 & moonrise_zkhost_mask_cosmos2025
ax2.annotate("$z_\\mathrm{phot}$" + f" wrongly outside 0.7$-$2.6 range: "
             + f"{100*np.sum(mask4)/len(cosmos2025_khost):.1f} \\%",
             xy=(0.05, 0.85), xycoords="axes fraction", fontsize=14,
             ha="left", va="top")

plt.savefig("cosmos2020_2025_photozs_vs_khostovan25_v1.1_speczs.pdf", dpi=300,
            bbox_inches="tight")
