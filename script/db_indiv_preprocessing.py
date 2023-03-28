#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.9

# @Time   : 2023/3/26 2:00
# @Author : 'Lou Zehua'
# @File   : db_indiv_preprocessing.py 

import os
import sys

if '__file__' not in globals():
    # !pip install ipynbname  # Remove comment symbols to solve the ModuleNotFoundError
    import ipynbname

    nb_path = ipynbname.path()
    __file__ = str(nb_path)
cur_dir = os.path.dirname(__file__)
pkg_rootdir = os.path.dirname(cur_dir)  # os.path.dirname()向上一级，注意要对应工程root路径
if pkg_rootdir not in sys.path:  # 解决ipynb引用上层路径中的模块时的ModuleNotFoundError问题
    sys.path.append(pkg_rootdir)
    print('-- Add root directory "{}" to system path.'.format(pkg_rootdir))

import re

import pandas as pd

encoding = "utf-8"


def key_uriform(key_common_name_str):
    key_common_name_str = str(key_common_name_str).lower().replace('.', '')  # remove '.'
    key_common_name_splits = re.split("[- /\():,]", key_common_name_str.lower())
    key_common_name_splits_striped = [s.strip() for s in key_common_name_splits if s.strip() != '']  # deal with the case: "Apache Jena - TDB"
    key_uriform = '-'.join(key_common_name_splits_striped)
    return key_uriform


def dbdbio_feat_preprocessing(src_path, tar_path, **kwargs):
    dtype = kwargs.get("dtype")
    df_dbdbio_info = pd.read_csv(src_path, encoding=encoding, index_col=False, dtype=dtype)
    # preprocessings
    pass
    # DBMS_uriform_recalc = df_dbdbio_info["card_title"].apply(key_uriform)
    # for i in range(len(DBMS_uriform_recalc)):
    #     if DBMS_uriform_recalc.values[i] != df_dbdbio_info["DBMS_uriform"].values[i]:
    #         print(DBMS_uriform_recalc.values[i], df_dbdbio_info["DBMS_uriform"].values[i])
    df_dbdbio_info.to_csv(tar_path, encoding=encoding, index=False)
    return


def dbengines_feat_preprocessing(src_path, tar_path, **kwargs):
    dtype = kwargs.get("dtype")
    df_dbengines_info = pd.read_csv(src_path, encoding=encoding, index_col=0, dtype=dtype)

    # preprocessings
    df_dbengines_info["DBMS_uriform"] = df_dbengines_info["DBMS"].apply(key_uriform)

    df_dbengines_info.to_csv(tar_path, encoding=encoding, index=False)
    return


if __name__ == '__main__':
    Base_Dir = pkg_rootdir
    src_dbdbio_dir = os.path.join(Base_Dir, "dbdbio_OSDB_info_crawling/data/manulabeled")
    src_dbengines_dir = os.path.join(Base_Dir, "db_engines_ranking_table_crawling/data/manulabeled")
    src_indiv_preprocessing_dir = os.path.join(Base_Dir, "data/db_indiv_preprocessing")

    month_yyyyMM = "202302"
    format_time_in_filename = "%Y%m"
    format_time_in_colname = "%b-%Y"
    from db_engines_ranking_table_crawling.script.time_format import TimeFormat

    curr_month = TimeFormat(month_yyyyMM, format_time_in_filename, format_time_in_filename)

    src_dbdbio_info_raw_path = os.path.join(src_dbdbio_dir, f"OSDB_info_{month_yyyyMM}_joined_manulabled.csv")
    src_dbengines_info_raw_path = os.path.join(src_dbengines_dir, f"ranking_crawling_{month_yyyyMM}_automerged_manulabled.csv")
    src_dbdbio_info_path = os.path.join(src_indiv_preprocessing_dir,
                                        f"OSDB_info_{month_yyyyMM}_joined_preprocessed.csv")
    src_dbengines_info_path = os.path.join(src_indiv_preprocessing_dir,
                                           f"ranking_crawling_{month_yyyyMM}_automerged_preprocessed.csv")

    dbdbio_info_dtype = {'Start Year': str, 'End Year': str}
    dbengines_info_dtype = {'initial_release_recalc': str, 'current_release_recalc': str, f'Rank_{curr_month.get_curr_month(format_time_in_colname)}': str}

    dbdbio_feat_preprocessing(src_dbdbio_info_raw_path, src_dbdbio_info_path, dtype=dbdbio_info_dtype)
    dbengines_feat_preprocessing(src_dbengines_info_raw_path, src_dbengines_info_path, dtype=dbengines_info_dtype)
