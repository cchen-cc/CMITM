import os
from pathlib import Path


DATA_BASE_DIR = '/home/local/PARTNERS/cu210/data_folder/cchen' 



# #############################################
# MIMIC-CXR-JPG constants
# #############################################
MIMIC_CXR_DATA_DIR = DATA_BASE_DIR + "/physionet.org/mimic/physionet.org/files/mimic-cxr-jpg/2.0.0"
# MIMIC_CXR_TRAIN_TXT = MIMIC_CXR_DATA_DIR / "train.txt"
# MIMIC_CXR_VALID_TXT = MIMIC_CXR_DATA_DIR / "test.txt"
MIMIC_CXR_CHEXPERT_CSV = MIMIC_CXR_DATA_DIR + "/mimic-cxr-2.0.0-chexpert.csv"
MIMIC_CXR_META_CSV = MIMIC_CXR_DATA_DIR + "/mimic-cxr-2.0.0-metadata.csv"
MIMIC_CXR_TEXT_CSV = MIMIC_CXR_DATA_DIR + "/mimic_cxr_sectioned.csv"
MIMIC_CXR_SPLIT_CSV = MIMIC_CXR_DATA_DIR + "/mimic-cxr-2.0.0-split.csv"
# Created csv
# MIMIC_CXR_TRAIN_CSV = MIMIC_CXR_DATA_DIR + "/train.csv"
MIMIC_CXR_TRAIN_CSV = MIMIC_CXR_DATA_DIR + "/train_new.csv"
# MIMIC_CXR_VALID_CSV = MIMIC_CXR_DATA_DIR + "/test.csv"
MIMIC_CXR_VALID_CSV = MIMIC_CXR_DATA_DIR + "/test_new.csv"
# MIMIC_CXR_TEST_CSV = MIMIC_CXR_DATA_DIR + "/test.csv"
MIMIC_CXR_TEST_CSV = MIMIC_CXR_DATA_DIR + "/test_new.csv"
# MIMIC_CXR_MASTER_CSV = MIMIC_CXR_DATA_DIR + "/master.csv"
MIMIC_CXR_MASTER_CSV = MIMIC_CXR_DATA_DIR + "/master_new.csv"
MIMIC_CXR_VIEW_COL = "ViewPosition"
MIMIC_CXR_PATH_COL = "Path"
MIMIC_CXR_SPLIT_COL = "split"