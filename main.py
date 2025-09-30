#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.9

# @Time   : 2023/3/15 0:44
# @Author : 'Lou Zehua'
# @File   : main.py 

import os
import shutil
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
from script.db_info_fusion import merge_key_dbdbio_dbengines, merge_info_dbdbio_dbengines, unique_name_recalc, \
    compare_df_curr_last_update_with_df_last_manulabeled_values

encoding = "utf-8"
Base_Dir = pkg_rootdir
src_dbdbio_dir = os.path.join(Base_Dir, "dbdbio_OSDB_info_crawling/data/manulabeled")
src_dbengines_dir = os.path.join(Base_Dir, "db_engines_ranking_table_crawling/data/manulabeled")
src_indiv_preprocessing_dir = os.path.join(Base_Dir, "data/db_indiv_preprocessing")
tar_dbfeatfusion_dir = os.path.join(Base_Dir, "data/db_feature_fusion")
mapping_table_dir = os.path.join(Base_Dir, "data/mapping_table")
manulabeled_dir = os.path.join(Base_Dir, "data/manulabeled")


if __name__ == '__main__':
    curr_stage = 2
    stage__DBNAME_MAPPING__RESET_FINAL_TABLE = {
        0: [False, True],
        # Resolve conflict for the dbname_mapping: src_dbfeatfusion_dbname_mapping_manulabeled_path before next step!
        1: [True, True],
        # Resolve conflict for the main part of final_table: dbfeatfusion_records_manulabeled_last_month_main_part_path before next step!
        2: [True, False]
    }

    month_YYYYmm = "202509"
    n_interval = 1
    format_time_in_filename = "%Y%m"
    format_time_in_colname = "%b-%Y"
    curr_month = TimeFormat(month_YYYYmm, format_time_in_filename, format_time_in_filename)
    last_month_yyyyMM = curr_month.get_last_month(format_time_in_filename, n=n_interval)
    colname_last_month = curr_month.get_last_month(format_time_in_colname, n=n_interval)
    colname_curr_month = curr_month.get_curr_month(format_time_in_colname)

    # Step1: preprocessing
    src_dbdbio_info_raw_path = os.path.join(src_dbdbio_dir, f"OSDB_info_{month_YYYYmm}_joined_manulabeled.csv")
    src_dbengines_info_raw_path = os.path.join(src_dbengines_dir, f"ranking_crawling_{month_YYYYmm}_automerged_manulabeled.csv")
    src_dbdbio_info_path = os.path.join(src_indiv_preprocessing_dir, f"OSDB_info_{month_YYYYmm}_joined_preprocessed.csv")
    src_dbengines_info_path = os.path.join(src_indiv_preprocessing_dir, f"ranking_crawling_{month_YYYYmm}_automerged_preprocessed.csv")

    dbdbio_info_dtype = {'Start Year': str, 'End Year': str}
    dbengines_info_dtype = {'initial_release_recalc': str, 'current_release_recalc': str, f'Rank_{colname_curr_month}': str}
    db_last_month_info_fusion_dtype = {'initial_release_year': str, 'current_release_year': str, f'Rank_{colname_last_month}': str}
    db_info_fusion_dtype = {'initial_release_year': str, 'current_release_year': str, f'Rank_{colname_curr_month}': str}

    dbdbio_feat_preprocessing(src_dbdbio_info_raw_path, src_dbdbio_info_path, dtype=dbdbio_info_dtype)
    dbengines_feat_preprocessing(src_dbengines_info_raw_path, src_dbengines_info_path, dtype=dbengines_info_dtype)

    # Step2: name alignment
    tar_dbfeatfusion_dbname_mapping_autogen_path = os.path.join(tar_dbfeatfusion_dir, f"dbfeatfusion_dbname_mapping_{month_YYYYmm}_autogen.csv")

    df_dbdbio_info = pd.read_csv(src_dbdbio_info_path, encoding=encoding, index_col=False, dtype=dbdbio_info_dtype)
    df_dbengines_info = pd.read_csv(src_dbengines_info_path, encoding=encoding, index_col=False, dtype=dbengines_info_dtype)
    # Filter open source projects
    opensource_license_filter_func = lambda x: str(x["open_source_license"]).startswith("Y")
    # Filter github open source projects
    has_github_repo_filter_func = lambda x: str(x["has_github_repo"]).startswith("Y") and str(x["has_github_repo"]).lower().find("_notgithub") < 0
    has_github_repo_or_opensource_filter_func = lambda x: opensource_license_filter_func(x) or has_github_repo_filter_func(x)
    filter_func = has_github_repo_or_opensource_filter_func
    df_dbdbio_info_ghos = df_dbdbio_info[df_dbdbio_info.apply(filter_func, axis=1)]
    df_dbengines_info_ghos = df_dbengines_info[df_dbengines_info.apply(filter_func, axis=1)]
    df_dbdbio_info_platform_filtered = df_dbdbio_info_ghos
    df_dbengines_info_platform_filtered = df_dbengines_info_ghos
    print(f"len_df_dbdbio_info_platform_filtered: {len(df_dbdbio_info_platform_filtered)}")
    print(f"len_df_dbengines_info_platform_filtered: {len(df_dbengines_info_platform_filtered)}")
    key_dbdbio_info, key_dbengines_info = "DBMS_urnform", "DBMS_urnform"
    key_db_info_pair = (key_dbdbio_info, key_dbengines_info)
    key_avoid_conf_prefixes = ("X_", "Y_")
    key_dbdbio_prefixed = key_avoid_conf_prefixes[0] + key_db_info_pair[0]
    key_dbengines_prefixed = key_avoid_conf_prefixes[1] + key_db_info_pair[1]
    merged_key_alias = "DBMS_urnform"
    match_state_field = "match_state"
    label_colname = "manu_labeled_flag"
    conflict_delimiter = "#dbdbio>|<dbengines#"

    src_dbfeatfusion_dbname_mapping_manulabeled_path = os.path.join(mapping_table_dir, f"dbfeatfusion_dbname_mapping_{month_YYYYmm}_manulabeled.csv")

    # Setp2.1: set False to init; Step2.2: set True to manulabel.
    DBNAME_MAPPING_CONFLICT_RESOLVED = stage__DBNAME_MAPPING__RESET_FINAL_TABLE[curr_stage][0]
    if not DBNAME_MAPPING_CONFLICT_RESOLVED:
        FORCE_INIT_DBFEATFUSION_DBNAME_MAPPING_MANULABELED = True
        # sort dataframes by keys before executing merge_key_dbdbio_dbengines
        df_dbdbio_info_platform_filtered = df_dbdbio_info_platform_filtered.sort_values(by=key_dbdbio_info, ascending=True)
        df_dbengines_info_platform_filtered = df_dbengines_info_platform_filtered.sort_values(by=key_dbengines_info, ascending=True)
        merge_key_dbdbio_dbengines(df_dbdbio_info_platform_filtered, df_dbengines_info_platform_filtered,
                                   save_path=tar_dbfeatfusion_dbname_mapping_autogen_path,
                                   on_key_pair=key_db_info_pair, key_avoid_conf_prefixes=key_avoid_conf_prefixes,
                                   merged_key_alias=merged_key_alias, match_state_field=match_state_field, label=label_colname)
        if FORCE_INIT_DBFEATFUSION_DBNAME_MAPPING_MANULABELED or not os.path.exists(src_dbfeatfusion_dbname_mapping_manulabeled_path):
            shutil.copyfile(src=tar_dbfeatfusion_dbname_mapping_autogen_path, dst=src_dbfeatfusion_dbname_mapping_manulabeled_path)
        print(f"{src_dbfeatfusion_dbname_mapping_manulabeled_path} initialized! \n\tPlease run main.py again after labeling it and setting DBNAME_MAPPING_CONFLICT_RESOLVED = True.")
    else:
        # manulabeled dbname_mapping
        state_error_msg = f"StateError! Please set DBNAME_MAPPING_CONFLICT_RESOLVED = False, then manually label the 'Fuzzy' and 'Multiple' Matched records in file {tar_dbfeatfusion_dbname_mapping_autogen_path}, " \
                          f"save it to the path: {src_dbfeatfusion_dbname_mapping_manulabeled_path}. Finally, set DBNAME_MAPPING_CONFLICT_RESOLVED = True."
        try:
            df_dbfeatfusion_dbname_mapping_manulabeled = pd.read_csv(src_dbfeatfusion_dbname_mapping_manulabeled_path, encoding=encoding, index_col=False, dtype=db_info_fusion_dtype)
            df_dbfeatfusion_dbname_mapping_manulabeled.drop(df_dbfeatfusion_dbname_mapping_manulabeled[df_dbfeatfusion_dbname_mapping_manulabeled['manu_labeled_flag'] == '--'].index, inplace=True)  # drop the merged redundant row marked the column 'manu_labeled_flag' as "--".
            df_dbfeatfusion_dbname_mapping_manulabeled[merged_key_alias] = df_dbfeatfusion_dbname_mapping_manulabeled.apply(
                lambda df: unique_name_recalc(df[key_dbdbio_prefixed], df[key_dbengines_prefixed]), axis=1)
            df_dbfeatfusion_dbname_mapping_manulabeled.to_csv(src_dbfeatfusion_dbname_mapping_manulabeled_path, encoding=encoding, index=False)
            # validate manulabeled
            assert(all(df_dbfeatfusion_dbname_mapping_manulabeled[label_colname].apply(lambda x: str(x).startswith("Y"))))
            assert(all(df_dbfeatfusion_dbname_mapping_manulabeled[match_state_field].apply(lambda x: str(x).split(':')[-1] in ["Normal", "X_Single", "Y_Single"])))
        except FileNotFoundError or AssertionError as e:
            raise Exception(state_error_msg)

        # Step3: DBMS features fusion
        settings_colnames_mapping_path = os.path.join(mapping_table_dir, "colnames_mapping.csv")
        df_settings_colnames_mapping = pd.read_csv(settings_colnames_mapping_path, encoding=encoding, index_col="tables")
        df_settings_colnames_mapping = df_settings_colnames_mapping.apply(lambda x: x.str.replace("COLNAME_CURR_MONTH", colname_curr_month))

        tar_dbfeatfusion_path = os.path.join(tar_dbfeatfusion_dir, f"dbfeatfusion_records_{month_YYYYmm}_automerged.csv")

        df_dbfeatfusion_dbname_mapping_manulabeled = pd.read_csv(src_dbfeatfusion_dbname_mapping_manulabeled_path, encoding=encoding, index_col=False)
        dbdbio_manulabed_flag_series = df_dbdbio_info_platform_filtered[key_dbdbio_info].apply(lambda x: x in df_dbfeatfusion_dbname_mapping_manulabeled[key_dbdbio_prefixed].values)
        dbengines_manulabed_flag_series = df_dbengines_info_platform_filtered[key_dbengines_info].apply(lambda x: x in df_dbfeatfusion_dbname_mapping_manulabeled[key_dbengines_prefixed].values)
        df_dbdbio_info_platform_filtered_manulabed = df_dbdbio_info_platform_filtered[dbdbio_manulabed_flag_series.values]
        df_dbengines_info_platform_filtered_manulabed = df_dbengines_info_platform_filtered[dbengines_manulabed_flag_series]

        merge_info_dbdbio_dbengines(df_dbdbio_info_platform_filtered_manulabed, df_dbengines_info_platform_filtered_manulabed,
                                    df_dbfeatfusion_dbname_mapping_manulabeled, save_path=tar_dbfeatfusion_path,
                                    df_feature_mapping=df_settings_colnames_mapping, input_key_colname="key",
                                    use_columns_merged=None, conflict_delimiter=conflict_delimiter, encoding=encoding)

        # Step4: Solve conflicts in the tar_dbfeatfusion_path manually.
        dbfeatfusion_records_automerged_last_month_path = os.path.join(tar_dbfeatfusion_dir, f"dbfeatfusion_records_{last_month_yyyyMM}_automerged.csv")
        dbfeatfusion_records_automerged_path = os.path.join(tar_dbfeatfusion_dir, f"dbfeatfusion_records_{month_YYYYmm}_automerged.csv")
        dbfeatfusion_records_manulabeled_last_month_path = os.path.join(manulabeled_dir, f"dbfeatfusion_records_{last_month_yyyyMM}_automerged_manulabeled.csv")
        tar_dbfeatfusion_records_manulabeled_path = os.path.join(manulabeled_dir, f"dbfeatfusion_records_{month_YYYYmm}_automerged_manulabeled.csv")

        #  Setp4.1: set True to init; Step4.2: set False to manulabel.
        RESET_FINAL_TABLE_TO_INHERIT_MANULABELED = stage__DBNAME_MAPPING__RESET_FINAL_TABLE[curr_stage][1]
        dbfeatfusion_records_manulabeled_last_month_main_part_path = dbfeatfusion_records_manulabeled_last_month_path.replace(".csv", "_main_part.csv")
        tar_dbfeatfusion_records_manulabeled_main_part_path = tar_dbfeatfusion_records_manulabeled_path.replace(".csv", "_main_part.csv")
        if RESET_FINAL_TABLE_TO_INHERIT_MANULABELED:
            df_last_automerged = pd.read_csv(dbfeatfusion_records_automerged_last_month_path, encoding=encoding, index_col=False, dtype=db_last_month_info_fusion_dtype)
            df_curr_automerged = pd.read_csv(dbfeatfusion_records_automerged_path, encoding=encoding, index_col=False, dtype=db_info_fusion_dtype)
            df_last_manulabeled = pd.read_csv(dbfeatfusion_records_manulabeled_last_month_path, encoding=encoding, index_col=False, dtype=db_last_month_info_fusion_dtype)
            curr_manulabeled_autogen_path = tar_dbfeatfusion_records_manulabeled_path
            valid_contains_conflict_delimiter = lambda x: str(x).find(conflict_delimiter) >= 0
            compare_df_curr_last_update_with_df_last_manulabeled_values(
                df_curr_automerged, df_last_automerged, df_last_manulabeled, save_path=curr_manulabeled_autogen_path,
                on_key_col="DBMS_urnform", ignore_cols=[f"Score_{colname_curr_month}", f"Rank_{colname_curr_month}"],
                index_filter_func=lambda x: not valid_contains_conflict_delimiter(x),
                item_filter_func=valid_contains_conflict_delimiter, encoding=encoding)
            print(f"{curr_manulabeled_autogen_path} initialized!")
            main_part_columns = df_last_manulabeled.columns[:-2]
            df_last_manulabeled[main_part_columns].to_csv(dbfeatfusion_records_manulabeled_last_month_main_part_path, encoding=encoding, index=False)
            print(dbfeatfusion_records_manulabeled_last_month_main_part_path, 'saved!')
            df_curr_manulabeled = pd.read_csv(tar_dbfeatfusion_records_manulabeled_path, encoding=encoding, index_col=False, dtype=db_info_fusion_dtype)
            df_curr_manulabeled[main_part_columns].to_csv(tar_dbfeatfusion_records_manulabeled_main_part_path, encoding=encoding, index=False)
            print(tar_dbfeatfusion_records_manulabeled_main_part_path, 'saved!')
            print(f"Warning: Please set RESET_FINAL_TABLE_TO_INHERIT_MANULABELED = False and rerun the code after manulabel the main part in {tar_dbfeatfusion_records_manulabeled_main_part_path}!")
        else:
            # run after manulabel the main part in tar_dbfeatfusion_records_manulabeled_main_part_path
            df_curr_manulabeled = pd.read_csv(tar_dbfeatfusion_records_manulabeled_path, encoding=encoding, index_col=False, dtype=db_info_fusion_dtype)
            df_curr_manulabeled_main_part = pd.read_csv(tar_dbfeatfusion_records_manulabeled_main_part_path, encoding=encoding, index_col=False, dtype=db_info_fusion_dtype)
            df_curr_manulabeled.update(df_curr_manulabeled_main_part)
            df_curr_manulabeled.to_csv(tar_dbfeatfusion_records_manulabeled_path, encoding=encoding, index=False)

