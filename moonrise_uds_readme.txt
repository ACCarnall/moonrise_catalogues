Column descriptions

MOONRISE_ID - as described in catalogue recipes guideline, catalogue index 06 added for GAIA stars
derek_uds_ID - Cosmos 2020 ID column
GAIA_STAR_ID - GAIA_DR3 source_id column
UDSZ_ID - Source ID where available from UDSz catalogue (https://www.nottingham.ac.uk/astronomy/UDS/UDSz/)
GALAMETZ18_ID - Source ID where available from Galametz et al. (2018) table 2
VANDELS_ID - ID taken from the VANDELS DR4 release
HIZ_ID - ID taken from Llerena high-z target catalogue
RA - icrs system, epoch 2016
DEC - icrs system, epoch 2016
PMRA - in mas/yr, taken from GAIA DR3 catalogue for stars, otherwise set to zero
PMDEC - in mas/yr, taken from GAIA DR3 catalogue for stars, otherwise set to zero
HMAG - H-band magnitude, either from Derek's catalogue VIDEO_H photometry, 2mass, or a polynomial fit to GAIA G-band magnitudes
HMAG_FLAG - 0 is HMAG from Derek's catalogue, 1 is HMAG from 2MASS PSC, 2 is HMAG derived from polynomial fit to GAIA G-band magnitudes
SIZE - arcsec, I had no information on this for UDS, so everything is set to -99
ZBEST - ZSPEC_DJA where available, else ZSPEC_VANDELS where available, else ZPHOT
ZBEST_FLAG - 0 if ZPHOT from Derek catalogue, 1 if udsz zspec, 2 if Galametz et al. (2018) zspec, 3 if VANDELS zspec, 4 if DJA zspec, 5 if zspec from high-z working group catalogue, 6 if zspec from AGN working group
ZPHOT - Derek ZPHOT (zphot column) - median value from 5 LePhare and 2 EAzY runs
ZSPEC_UDSZ - Spectroscopic redshift where available from UDSz (https://www.nottingham.ac.uk/astronomy/UDS/UDSz/)
ZFLAG_UDSZ - Spectroscopic redshift flag from UDSz catalogue
ZSPEC_GALAMETZ18 - Spectroscopic redshift where available from Galametz et al. (2018) table 2
ZFLAG_GALAMETZ18 - Spectroscopic redshift flag where available from Galametz et al. (2018) table 2
ZSPEC_VANDELS - Spectroscopic redshift where available from VANDELS DR4 release, only included for zflags 3, 4, 9, 13, 14, 19, 23, 24, 29, 214
ZFLAG_VANDELS - spectroscopic redshift flag taken from the VANDELS DR4 release
ZSPEC_DJA - DJA specz, derived as follows: first cut DJA csv v4.5 catalogue to zgrade=3, then perform internal match within 0.3" marking groups of rows. Then within each group of rows calculate the mean ra, dec, zfit and the standard deviation on zfit. Then drop rows with standard deviation on zfit > 0.05 (objects with multiple confident redshifts that diagree with each other), then join this subsample with objects that were not part of a group (i.e., objects that only had one spectrum with zgrade=3)
AGN_ZSPEC - ZSPEC_OTHER column taken from AGN catalogue (all available zspec values for AGN?)
DJA_SOURCE_ID - source ID listed by DJA, note these are not necessarily unique (only populated if DJA had only 1 spectrum for this object)
DJA_PROG_ID - ID of the programme the DJA spectrum came from (only populated if DJA had only 1 spectrum for this object)
STAR - Denotes GAIA stars: 1 = Star from the GAIA catalogue, 0 = all other objects
GOOD_STAR - GAIA star for potential use as acquisition star, selected by: GAIA DR3 ruwe below Castro-Ginard et al. (2024) threshold & GAIA DR3 phot_variable_flag != VARIABLE & GAIA DR3 non_single_star == 0 & GAIA DR3 pmra < 0.1 arcsec/yr & GAIA DR3 pmdec < 0.1 arcsec/yr & HMAG_FLAG <= 1
RUWE - GAIA DR3 ruwe column
GAIA_magG - GAIA DR3 phot_g_mean_mag column
GAIA_magR - GAIA DR3 phot_rp_mean_mag column
ABS_U - Absolute U-band magnitude from bagpipes fitting
ABS_V - Absolute V-band magnitude from bagpipes fitting
ABS_J - Absolute J-band magnitude from bagpipes fitting
stellar_mass_16 - 16th percentile of stellar mass posterior from bagpipes fitting
stellar_mass_50 - 50th percentile of stellar mass posterior from bagpipes fitting
stellar_mass_84 - 84th percentile of stellar mass posterior from bagpipes fitting
AGN_texp - (hours) highest of the three values in the AGN working group's "Exposure_" columns - only objects with a positive value in any of these columns were propagated to this catalogue
AGN_ID_OTHER_REF - ID_OTHER_REF column taken from AGN catalogue (Ref and ID for AGN_ZPEC?)
AGN_FLAG_MAG - FLAG_MAG column taken from AGN catalogue (source for HMAG*0.9 value used in this catalogue)
MW_EBV - Milky Way E(B-V) value taken from dustmaps package
in_passive - moonrise passive sample flag
in_starforming - moonrise star-forming sample flag
in_AGN - moonrise agn sample flag
in_highz - moonrise highz sample flag