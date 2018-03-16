#!/usr/bin/env python
"""
Download the data to compute shortwave cloud forcing from the set of
10 CMIP5 models participating in the sstClim/sstClimAerosol experiments

"""
from CEDA_download import *
import logging

SAVE_PATH = "cmip5_tj/{experiment}/{model}"
OVERWRITE = True

set_logger(filename="cmip5_aie.log")

#username, password = get_credentials()
username = 'tvandal'
password = 'Snow4713'
group_models = [
#    ("BCC", "bcc-csm1-1"),
#    ("CCCma", "CanESM2"),
#    ("CSIRO-QCCCE", "CSIRO-Mk3-6-0"),
#    ("IPSL", "IPSL-CM5A-LR"),
#    ("LASG-IAP", "FGOALS-s2"),
#    ("MIROC", "MIROC5"),
#    ("MOHC", "HadGEM2-A"),
    ("MRI", "MRI-CGCM3"),
#    ("NCC", "NorESM1-M"),
#    ("NOAA-GFDL", "GFDL-CM3"),
]

experiments = ["rcp85"]
freqs = ["day", ]
ensembles = ["r1i1p1", ]

########################################################################

# Set 1 - atmosphere/radiation properties
logging.info("--- BEGIN Set 1")
realms = ["atmos", ]
cmor_tables = ["day", ]
variables = ["tasmax", 'tasmin', 'pr']

datasets = get_datasets(group_models, experiments, freqs, realms,
                        cmor_tables, ensembles, variables)
download_batch(datasets, SAVE_PATH,
               username, password, overwrite=OVERWRITE)
logging.info("--- END Set 1")
