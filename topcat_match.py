import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.table import Table
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy import units as u


def topcat_symmetric_match(incat1, incat2, dist_arcsec, join_type="1 and 2",
                           ra_col_1="ra", dec_col_1="dec",
                           ra_col_2="ra", dec_col_2="dec",
                           suffix1="_1", suffix2="_2"):
    """
    Cross-match two catalogues based on ra and dec within some tolerance.
    Reproduces functionality of TOPCAT tool's sky match algorithm with
    the match selection choice "best match, symmetric".
    """

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
                                   result1[1][dist_mask1].value],
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
        cat1 = incat1.copy()
        cat2 = incat2.copy()
        swap = False
    else:
        dist_mask = dist_mask2
        idxmin = idxmin2
        result = result2
        cat1 = incat2.copy()
        cat2 = incat1.copy()
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

    # Merge in columns from cat2 into rows they matched with in cat1
    # At this point, each cat2 row can be matched to multiple cat1 rows
    matched_dupes = pd.merge(cat1[dist_mask].reset_index(drop=True),
                             cat2.iloc[indices_cat2].reset_index(drop=True),
                             left_index=True, right_index=True, how="inner",
                             suffixes=(suffix1, suffix2))

    # Add columns for the index of the matched cat2 row and match separation
    matched_dupes["cat2_row"] = indices_cat2
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
        mask = ~cat2.index.isin(result[0][dist_mask])
        to_concat.append(cat2.iloc[mask].rename(columns=rename_cols))

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

    out = positional_cross_match(khost, cosmos2025_fluxes, dist_arcsec=0.5,
                                ra_col_1="ra", dec_col_1="dec",
                                ra_col_2="ra", dec_col_2="dec",
                                suffix1="_khost", suffix2="_c2025",
                                join_type="1 and 2")

    print(out.shape)
    print(out.columns)
    print(out.columns.drop_duplicates())

    #Table.from_pandas(out).write("test.fits", overwrite=True)
