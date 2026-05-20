import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.table import Table
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy import units as u


def pair_match_sky(incat1, incat2, dist_arcsec, join_type="1 and 2",
                   match_selection="Best match, symmetric",
                   ra_col_1="ra", dec_col_1="dec",
                   ra_col_2="ra", dec_col_2="dec",
                   suffix1="_1", suffix2="_2"):

    """
    Cross-match two catalogues based on ra and dec within some tolerance.
    Reproduces as closely as possible TOPCAT's pair sky match algorithm.
    """

    if match_selection == "Best match, symmetric":
        return pair_match_sky_symmetric(incat1, incat2, dist_arcsec,
                                        join_type=join_type,
                                        ra_col_1=ra_col_1, dec_col_1=dec_col_1,
                                        ra_col_2=ra_col_2, dec_col_2=dec_col_2,
                                        suffix1=suffix1, suffix2=suffix2)

    if match_selection == "All matches":
        return pair_match_sky_all(incat1, incat2, dist_arcsec,
                                  join_type=join_type,
                                  ra_col_1=ra_col_1, dec_col_1=dec_col_1,
                                  ra_col_2=ra_col_2, dec_col_2=dec_col_2,
                                  suffix1=suffix1, suffix2=suffix2)


def pair_match_sky_symmetric(incat1, incat2, dist_arcsec,
                             join_type="1 and 2",
                             ra_col_1="ra", dec_col_1="dec",
                             ra_col_2="ra", dec_col_2="dec",
                             suffix1="_1", suffix2="_2"):

    coords1 = SkyCoord(ra=incat1[ra_col_1].values*u.degree,
                       dec=incat1[dec_col_1].values*u.degree)
    coords2 = SkyCoord(ra=incat2[ra_col_2].values*u.degree,
                       dec=incat2[dec_col_2].values*u.degree)

    # result[0] is index of closest match in coords2 for each coords1 source
    # result[1] is sep between each coords1 source and closest match in coords2
    result1 = match_coordinates_sky(coords1, coords2)
    result2 = match_coordinates_sky(coords2, coords1)

    # create mask for only matches within dist_arcsec separation threshold
    dist_mask1 = result1[1] < dist_arcsec*u.arcsec
    dist_mask2 = result2[1] < dist_arcsec*u.arcsec

    result1_df = pd.DataFrame(np.c_[result1[0][dist_mask1],
                                   result1[1][dist_mask1].arcsec],
                             columns=["cat2_row", "match_sep_arcsec"])

    result2_df = pd.DataFrame(np.c_[result2[0][dist_mask2],
                                   result2[1][dist_mask2].value],
                             columns=["cat1_row", "match_sep_arcsec"])

    idxmin1 = result1_df.groupby("cat2_row")["match_sep_arcsec"].idxmin()
    idxmin2 = result2_df.groupby("cat1_row")["match_sep_arcsec"].idxmin()

    # Pick which way round to do merge based on which has more matches
    # Seems to be what TOPCAT does for "best match, symmetric" option
    if len(idxmin1) > len(idxmin2):
        dist_mask = dist_mask1
        idxmin = idxmin1
        result = result1
        cat1 = incat1.copy().reset_index(drop=True)
        cat2 = incat2.copy().reset_index(drop=True)
        swap = False
    else:
        dist_mask = dist_mask2
        idxmin = idxmin2
        result = result2
        cat1 = incat2.copy().reset_index(drop=True)
        cat2 = incat1.copy().reset_index(drop=True)
        swap = True

    if swap:
        if join_type == "All from 1":
            join_type = "All from 2"
        elif join_type == "All from 2":
            join_type = "All from 1"
        elif join_type == "1 not 2":
            join_type = "2 not 1"
        elif join_type == "2 not 1":
            join_type = "1 not 2"
        suffix1, suffix2 = suffix2, suffix1

    indices_cat2 = result[0][dist_mask]
    unmatched_cat2 = ~cat2.index.isin(indices_cat2)

    # Merge in columns from cat2 into rows they matched with in cat1
    # At this point, each cat2 row can be matched to multiple cat1 rows
    matched_dupes = pd.merge(cat1[dist_mask].reset_index(drop=True),
                             cat2.iloc[indices_cat2].reset_index(drop=True),
                             left_index=True, right_index=True, how="inner",
                             suffixes=(suffix1, suffix2))

    # Add columns for the index of the matched cat2 row and match separation
    matched_dupes["match_sep_arcsec"] = result[1][dist_mask]

    # If multiple cat1 rows match to the same cat2 row,
    # keep only the cat1 row with the smallest separation
    matched = matched_dupes.loc[idxmin].reset_index(drop=True)

    to_concat = []
    if join_type in ["1 and 2", "1 or 2", "All from 1", "All from 2"]:
        to_concat.append(matched)
    if join_type in ["All from 1", "1 or 2", "1 not 2"]:
        # Add in unmatched rows from cat1
        rename_cols = cat1.columns[np.isin(cat1.columns, cat2.columns)]
        rename_cols = {col: col+suffix1 for col in rename_cols}
        to_concat.append(cat1[~dist_mask].rename(columns=rename_cols))

        mask = ~np.isin(matched_dupes.index, idxmin)
        to_concat.append(matched_dupes[mask].rename(columns=rename_cols))

    if join_type in ["All from 2", "1 or 2", "2 not 1"]:
        # Add in unmatched rows from cat2
        rename_cols = cat2.columns[np.isin(cat2.columns, cat1.columns)]
        rename_cols = {col: col+suffix2 for col in rename_cols}
        to_concat.append(cat2[unmatched_cat2].rename(columns=rename_cols))

    if len(to_concat) > 1:
        matched = pd.concat(to_concat, ignore_index=True)

    elif len(to_concat) == 1:
        matched = to_concat[0]

    return matched.reset_index(drop=True)


def pair_match_sky_all(incat1, incat2, dist_arcsec, join_type="1 and 2",
                       ra_col_1="ra", dec_col_1="dec",
                       ra_col_2="ra", dec_col_2="dec",
                       suffix1="_1", suffix2="_2"):

    coords1 = SkyCoord(ra=incat1[ra_col_1].values*u.degree,
                       dec=incat1[dec_col_1].values*u.degree)
    coords2 = SkyCoord(ra=incat2[ra_col_2].values*u.degree,
                       dec=incat2[dec_col_2].values*u.degree)

    search = coords2.search_around_sky(coords1, dist_arcsec*u.arcsec)
    idx_cat1, idx_cat2, d2d, d3d = search

    cat1 = incat1.copy()
    cat2 = incat2.copy()

    # Rename columns that are in both cat1 and cat2 to avoid merge conflicts
    cat1_cols_replace = cat1.columns[cat1.columns.isin(cat2.columns)]
    cat2_cols_replace = cat2.columns[cat2.columns.isin(cat1.columns)]

    cat1.rename(columns={col: col+suffix1 for col in cat1_cols_replace},
                inplace=True)
    cat2.rename(columns={col: col+suffix2 for col in cat2_cols_replace},
                inplace=True)

    # tile input catalogues according to their repeating indices in idx_cat1+2
    cat1_dupe = cat1.iloc[idx_cat1, :]
    cat2_dupe = cat2.iloc[idx_cat2, :]

    cat1_dupe["match_sep_arcsec"] = d2d.arcsec
    # merge the two
    matched = pd.concat([cat1_dupe.reset_index(drop=True),
                         cat2_dupe.reset_index(drop=True)], axis=1)

    unmatched_cat1 = cat1[~cat1.index.isin(idx_cat1)]
    unmatched_cat2 = cat2[~cat2.index.isin(idx_cat2)]

    to_concat = []
    if join_type in ["1 and 2", "1 or 2", "All from 1", "All from 2"]:
        to_concat.append(matched)
    if join_type in ["All from 1", "1 or 2", "1 not 2"]:
        to_concat.append(unmatched_cat1)

    if join_type in ["All from 2", "1 or 2", "2 not 1"]:
        # Add in unmatched rows from cat2
        to_concat.append(unmatched_cat2)

    if len(to_concat) > 1:
        matched = pd.concat(to_concat, ignore_index=True)

    elif len(to_concat) == 1:
        matched = to_concat[0]

    return matched.reset_index(drop=True)


if __name__ == "__main__":
    # TESTING

    # COSMOS2025 photometric catalogue
    c2025_flux_path = "COSMOSWeb_mastercatalog_v1.1_photom_primary.fits"
    cosmos2025_fluxes = Table.read(c2025_flux_path)

    cosmos2025_fluxes = cosmos2025_fluxes[["id", "ra", "dec"]]
    cosmos2025_fluxes = cosmos2025_fluxes.to_pandas()

    # Khostovan et al. (2025) v1.1 specz compilation
    specz_path = "specz_compilation_COSMOS_DR1.1_unique.fits"
    khost = Table.read(specz_path).to_pandas()
    khost.rename(columns={"specz": "specz_khost", "ra_corrected": "ra",
                        "dec_corrected": "dec"}, inplace=True)

    khost = khost[["Id_specz", "ra", "dec"]]

    # DJA v4.5 NIRSpec catalogue
    dja_cat = pd.read_csv("dja_nirspec_v4.5.csv")

    gaia_table = Table.read("gaia_stars_cosmos.fits").to_pandas()
    ra_mask = (gaia_table["ra"] > 149.05) & (gaia_table["ra"] < 151.07)
    dec_mask = (gaia_table["dec"] > 1.39) & (gaia_table["dec"] < 3.08)
    gaia_table = gaia_table[ra_mask & dec_mask]

    cosmos2020 = Table.read("COSMOS2020_reduced_columns.fits").to_pandas()

    print(gaia_table.shape, cosmos2020.shape)
    out = pair_match_sky(gaia_table, cosmos2020, dist_arcsec=0.5,
                                ra_col_1="ra", dec_col_1="dec",
                                ra_col_2="ALPHA_J2000", dec_col_2="DELTA_J2000",
                                suffix1="_gaia", suffix2="_c2020",
                                join_type="All from 1",
                                match_selection="Best match, symmetric")

    print(out.shape)

    Table.from_pandas(out).write("test.fits", overwrite=True)
