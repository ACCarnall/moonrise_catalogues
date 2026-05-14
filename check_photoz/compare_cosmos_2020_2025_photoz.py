import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.table import Table
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy import units as u


def positional_cross_match(cat1, cat2, dist_arcsec):
    """
    Cross-match two catalogues based on their RA and DEC columns, returning
    a new joined catalogue with only the closest matched sources.
    """

    # Create SkyCoord objects for both catalogues
    coords1 = SkyCoord(ra=cat1['ra'].values*u.degree,
                       dec=cat1['dec'].values*u.degree)
    coords2 = SkyCoord(ra=cat2['ra'].values*u.degree,
                       dec=cat2['dec'].values*u.degree)

    # Perform cross-matching using astropy's search_around_sky function
    result = match_coordinates_sky(coords1, coords2)

    dist_mask = result[1] < dist_arcsec*u.arcsec

    matched = pd.merge(cat1[dist_mask].reset_index(drop=True),
                       cat2.iloc[result[0][dist_mask]].reset_index(drop=True),
                       left_index=True, right_index=True,
                       suffixes=('_1', '_2'))

    return matched


# ##### Load up base tables #####

# COSMOS2020 catalogue, unedited download from
# https://irsa.ipac.caltech.edu/data/COSMOS/tables/cosmos2020/

cosmos2020 = Table.read("../COSMOS2020_CLASSIC_R1_v2.2_p3.fits").to_pandas()

# Cuts to exactly reproduce Ross's 2023 sample selection that was used to
# define objects Bagpipes was run on for UVJ colours and stellar masses
cosmos2020 = cosmos2020[cosmos2020["FLAG_COMBINED"] == 0]
cosmos2020 = cosmos2020[cosmos2020["UVISTA_H_MAG_APER3"] <= 24]

cosmos2020.rename(columns={"ALPHA_J2000": "ra",
                           "DELTA_J2000": "dec"}, inplace=True)

# COSMOS2025 catalogue, unedited download from
# https://cosmos2025.iap.fr/catalog_download.php

c2025_flux_path = "../COSMOSWeb_mastercatalog_v1.1_photom_primary.fits"
cosmos2025_fluxes = Table.read(c2025_flux_path)#.to_pandas()

aperture_diameters = [0.2, 0.3, 0.5, 0.75]
aper_cols = [col for col in cosmos2025_fluxes.colnames if "_aper_" in col]

for aper_col in aper_cols:
    for i in range(len(aperture_diameters)):
        diam = aperture_diameters[i]
        new_col_name = aper_col + f"_{diam}_arcsec"
        cosmos2025_fluxes[new_col_name] = cosmos2025_fluxes[aper_col][:, i]

    del cosmos2025_fluxes[aper_col]

cosmos2025_fluxes = cosmos2025_fluxes.to_pandas()

c2025_photoz_path = "../COSMOSWeb_mastercatalog_v1.1_lephare.fits"
cosmos2025_photoz = Table.read(c2025_photoz_path).to_pandas()

cosmos2025 = pd.merge(cosmos2025_fluxes, cosmos2025_photoz, left_index=True,
                      right_index=True)

cosmos2025 = cosmos2025[cosmos2025["mag_auto_f115w"] < 24]
cosmos2025 = cosmos2025[cosmos2025["zfinal"] > 0]


# DJA v4.5 NIRSpec catalogue, unedited csv download from
# https://s3.amazonaws.com/msaexp-nirspec/extractions/nirspec_public_v4.5.html
dja_cat = pd.read_csv("../dja_nirspec_v4.5.csv")
dja_cat = dja_cat[dja_cat["grade"] == 3]



# Match and plot
cosmos2020_dja = positional_cross_match(cosmos2020, dja_cat, 0.35)
cosmos2025_dja = positional_cross_match(cosmos2025, dja_cat, 0.35)

moonrise_zphot_mask_cosmos2020 = ((cosmos2020_dja["ez_z_phot"] >= 0.7)
                                  & (cosmos2020_dja["ez_z_phot"] <= 2.6))

moonrise_zphot_mask_cosmos2025 = ((cosmos2025_dja["zfinal"] >= 0.7)
                                  & (cosmos2025_dja["zfinal"] <= 2.6))

moonrise_zdja_mask_cosmos2020 = ((cosmos2020_dja["zfit"] >= 0.7)
                                 & (cosmos2020_dja["zfit"] <= 2.6))

moonrise_zdja_mask_cosmos2025 = ((cosmos2025_dja["zfit"] >= 0.7)
                                 & (cosmos2025_dja["zfit"] <= 2.6))

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

#plt.show()
#plt.close()
plt.savefig("cosmos2020_2025_dja_comparison.pdf", dpi=300,
            bbox_inches="tight")



# khost et al. (2025) specz compilation, unedited download from
# https://github.com/cosmosastro/speczcompilation
specz_path = "../specz_compilation_COSMOS_DR1.1_unique.fits"
khost = Table.read(specz_path).to_pandas()
khost.rename(columns={"specz": "specz_khost", "ra_corrected": "ra",
                          "dec_corrected": "dec"}, inplace=True)

khost = khost[khost["Confidence_level"] >= 97]
#khost.drop_duplicates(subset="Id_COS20_Classic", inplace=True)

khost.index = khost["Id_COS20_Classic"].values

cosmos2020_khost = pd.merge(cosmos2020, khost, left_on="ID",
                                right_on="Id_COS20_Classic",
                                how="inner", suffixes=("_1", "_2"))

cosmos2025_khost = pd.merge(cosmos2025, khost, left_on="id",
                                right_on="Id_COSMOS25",
                                how="inner", suffixes=("_1", "_2"))

#cosmos2020_khost = positional_cross_match(cosmos2020, khost, 0.3)
#cosmos2025_khost = positional_cross_match(cosmos2025, khost, 0.3)

moonrise_zphot_mask_cosmos2020 = ((cosmos2020_khost["ez_z_phot"] >= 0.7)
                                  & (cosmos2020_khost["ez_z_phot"] <= 2.6))

moonrise_zphot_mask_cosmos2025 = ((cosmos2025_khost["zfinal"] >= 0.7)
                                  & (cosmos2025_khost["zfinal"] <= 2.6))

moonrise_zkhost_mask_cosmos2020 = ((cosmos2020_khost["specz_khost"] >= 0.7)
                                 & (cosmos2020_khost["specz_khost"] <= 2.6))

moonrise_zkhost_mask_cosmos2025 = ((cosmos2025_khost["specz_khost"] >= 0.7)
                                 & (cosmos2025_khost["specz_khost"] <= 2.6))

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

#plt.show()
#plt.close()
plt.savefig("cosmos2020_2025_khostovan25_comparison.pdf", dpi=300,
            bbox_inches="tight")
