#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.9

# @Time   : 2023/3/15 0:44
# @Author : 'Lou Zehua'
# @File   : main.py 

import os
import sys

if '__file__' not in globals():
    # !pip install ipynbname  # Remove comment symbols to solve the ModuleNotFoundError
    import ipynbname

    nb_path = ipynbname.path()
    __file__ = str(nb_path)
cur_dir = os.path.dirname(__file__)
pkg_rootdir = cur_dir  # os.path.dirname()向上一级，注意要对应工程root路径
if pkg_rootdir not in sys.path:  # 解决ipynb引用上层路径中的模块时的ModuleNotFoundError问题
    sys.path.append(pkg_rootdir)
    print('-- Add root directory "{}" to system path.'.format(pkg_rootdir))

import pandas as pd

from db_engines_ranking_table_crawling.script.time_format import TimeFormat
from script.db_indiv_preprocessing import dbdbio_feat_preprocessing, dbengines_feat_preprocessing
from script.db_info_fusion import merge_key_dbdbio_dbengines, merge_info_dbdbio_dbengines, unique_name_recalc

encoding = "utf-8"
Base_Dir = pkg_rootdir
src_dbdbio_dir = os.path.join(Base_Dir, "dbdbio_OSDB_info_crawling/data/manulabeled")
src_dbengines_dir = os.path.join(Base_Dir, "db_engines_ranking_table_crawling/data/manulabeled")
src_indiv_preprocessing_dir = os.path.join(Base_Dir, "data/db_indiv_preprocessing")
tar_dbfeatfusion_dir = os.path.join(Base_Dir, "data/db_feature_fusion")


if __name__ == '__main__':
    month_yyyyMM = "202302"
    format_time_in_filename = "%Y%m"
    format_time_in_colname = "%b-%Y"
    curr_month = TimeFormat(month_yyyyMM, format_time_in_filename, format_time_in_filename)

    # Step1: preprocessing
    src_dbdbio_info_raw_path = os.path.join(src_dbdbio_dir, f"OSDB_info_{month_yyyyMM}_joined_manulabled.csv")
    src_dbengines_info_raw_path = os.path.join(src_dbengines_dir, f"ranking_crawling_{month_yyyyMM}_automerged_manulabled.csv")
    src_dbdbio_info_path = os.path.join(src_indiv_preprocessing_dir, f"OSDB_info_{month_yyyyMM}_joined_preprocessed.csv")
    src_dbengines_info_path = os.path.join(src_indiv_preprocessing_dir, f"ranking_crawling_{month_yyyyMM}_automerged_preprocessed.csv")

    dbdbio_info_dtype = {'Start Year': str, 'End Year': str}
    dbengines_info_dtype = {'initial_release_recalc': str, 'current_release_recalc': str, f'Rank_{curr_month.get_curr_month(format_time_in_colname)}': str}

    dbdbio_feat_preprocessing(src_dbdbio_info_raw_path, src_dbdbio_info_path, dtype=dbdbio_info_dtype)
    dbengines_feat_preprocessing(src_dbengines_info_raw_path, src_dbengines_info_path, dtype=dbengines_info_dtype)

    # Step2: name alignment
    tar_dbfeatfusion_dbname_mapping_autogen_path = os.path.join(tar_dbfeatfusion_dir, f"dbfeatfusion_dbname_mapping_{month_yyyyMM}_autogen.csv")

    df_dbdbio_info = pd.read_csv(src_dbdbio_info_path, encoding=encoding, index_col=False, dtype=dbdbio_info_dtype)
    df_dbengines_info = pd.read_csv(src_dbengines_info_path, encoding=encoding, index_col=False, dtype=dbengines_info_dtype)
    # Filter github open source projects
    filter_func = lambda x: str(x).startswith("Y")
    df_dbdbio_info_ghos = df_dbdbio_info[df_dbdbio_info["has_open_source_github_repo"].apply(filter_func)]
    df_dbengines_info_ghos = df_dbengines_info[df_dbengines_info["has_open_source_github_repo"].apply(filter_func)]
    df_dbdbio_info_platform_filtered = df_dbdbio_info_ghos
    df_dbengines_info_platform_filtered = df_dbengines_info_ghos
    print(f"len_df_dbdbio_info_platform_filtered: {len(df_dbdbio_info_platform_filtered)}")
    print(f"len_df_dbengines_info_platform_filtered: {len(df_dbengines_info_platform_filtered)}")
    key_dbdbio_info, key_dbengines_info = "DBMS_uriform", "DBMS_uriform"
    key_db_info_pair = (key_dbdbio_info, key_dbengines_info)
    key_avoid_conf_prefixes = ("X_", "Y_")
    key_dbdbio_prefixed = key_avoid_conf_prefixes[0] + key_db_info_pair[0]
    key_dbengines_prefixed = key_avoid_conf_prefixes[1] + key_db_info_pair[1]
    merged_key_alias = "DBMS_uriform"
    match_state_field = "match_state"
    label_colname = "manu_labeled_flag"
    merge_key_dbdbio_dbengines(df_dbdbio_info_platform_filtered, df_dbengines_info_platform_filtered,
                               save_path=tar_dbfeatfusion_dbname_mapping_autogen_path,
                               on_key_pair=key_db_info_pair, key_avoid_conf_prefixes=key_avoid_conf_prefixes,
                               merged_key_alias=merged_key_alias, match_state_field=match_state_field, label=label_colname)

    CONFLICT_RESOLVED = True
    # manulabeled dbname_mapping
    src_dbfeatfusion_dbname_mapping_manulabeled_path = os.path.join(Base_Dir, f"data/mapping_table/dbfeatfusion_dbname_mapping_{month_yyyyMM}_manulabeled.csv")
    if CONFLICT_RESOLVED:
        df_dbfeatfusion_dbname_mapping_manulabeled = pd.read_csv(src_dbfeatfusion_dbname_mapping_manulabeled_path, encoding=encoding, index_col=False)
        df_dbfeatfusion_dbname_mapping_manulabeled[merged_key_alias] = df_dbfeatfusion_dbname_mapping_manulabeled.apply(
            lambda df: unique_name_recalc(df[key_dbdbio_prefixed], df[key_dbengines_prefixed]), axis=1)
        df_dbfeatfusion_dbname_mapping_manulabeled.to_csv(src_dbfeatfusion_dbname_mapping_manulabeled_path, encoding=encoding, index=False)
        # validate manulabeled
        assert(all(df_dbfeatfusion_dbname_mapping_manulabeled[label_colname].apply(lambda x: str(x).startswith("Y"))))
        assert(all(df_dbfeatfusion_dbname_mapping_manulabeled[match_state_field].apply(lambda x: str(x).split(':')[-1] in ["Normal", "X_Single", "Y_Single"])))
    else:
        raise ValueError(f"StateError! Please manually label the 'Fuzzy' and 'Multiple' Matched records in file {tar_dbfeatfusion_dbname_mapping_autogen_path}, "
                         f"save it to the path: {src_dbfeatfusion_dbname_mapping_manulabeled_path}, and set CONFLICT_RESOLVED = True.")

    # Step3: DBMS features fusion
    settings_colnames_mapping_path = os.path.join(Base_Dir, "data/mapping_table/colnames_mapping.csv")
    df_settings_colnames_mapping = pd.read_csv(settings_colnames_mapping_path, encoding=encoding, index_col="tables")

    tar_dbfeatfusion_path = os.path.join(tar_dbfeatfusion_dir, f"dbfeatfusion_records_{month_yyyyMM}_automerged.csv")

    df_dbfeatfusion_dbname_mapping_manulabeled = pd.read_csv(src_dbfeatfusion_dbname_mapping_manulabeled_path, encoding=encoding, index_col=False)
    dbdbio_manulabed_flag_series = df_dbdbio_info_platform_filtered[key_dbdbio_info].apply(lambda x: x in df_dbfeatfusion_dbname_mapping_manulabeled[key_dbdbio_prefixed].values)
    dbengines_manulabed_flag_series = df_dbengines_info_platform_filtered[key_dbengines_info].apply(lambda x: x in df_dbfeatfusion_dbname_mapping_manulabeled[key_dbengines_prefixed].values)
    df_dbdbio_info_platform_filtered_manulabed = df_dbdbio_info_platform_filtered[dbdbio_manulabed_flag_series.values]
    df_dbengines_info_platform_filtered_manulabed = df_dbengines_info_platform_filtered[dbengines_manulabed_flag_series]

    merge_info_dbdbio_dbengines(df_dbdbio_info_platform_filtered_manulabed, df_dbengines_info_platform_filtered_manulabed,
                                df_dbfeatfusion_dbname_mapping_manulabeled, save_path=tar_dbfeatfusion_path,
                                df_feature_mapping=df_settings_colnames_mapping, input_key_colname="key",
                                use_columns_merged=None, encoding=encoding)
    # Step4: Solve conflicts in the tar_dbfeatfusion_path manually.
