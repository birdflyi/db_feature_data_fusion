#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.9

# @Time   : 2023/3/16 16:13
# @Author : 'Lou Zehua'
# @File   : db_info_fusion.py

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

import copy
import numpy as np
import pandas as pd

from db_engines_ranking_table_crawling.script.time_format import TimeFormat

from script.db_indiv_preprocessing import key_uriform


def uriform_validate(str_list, ignore_na=False):
    str_list = list(str_list)
    flag = True
    for elem in str_list:
        if pd.isna(elem):
            if ignore_na:
                pass
            else:
                print("NAN is not a uriform!")
                flag = False
                break
        else:
            elem_key_uriform = key_uriform(elem)
            flag = flag and (str(elem) == str(elem_key_uriform))
    return flag


def merge_key_dbdbio_dbengines(df1, df2, save_path, on_key_pair, key_avoid_conf_prefixes=("X_", "Y_"),
                               merged_key_alias="index", match_state_field="match_state", label="manu_labeled_flag",
                               encoding="utf-8"):
    on_key_pair = on_key_pair or ("Name", "Name")
    key_df1, key_df2 = on_key_pair[:2]
    assert(len(df1) == len(set(df1[key_df1].values)))
    assert(len(df2) == len(set(df2[key_df2].values)))

    dbdbio_unique_name = key_avoid_conf_prefixes[0] + key_df1
    dbengines_unique_name = key_avoid_conf_prefixes[1] + key_df2

    init_d = {
        merged_key_alias: None,
        dbdbio_unique_name: None,
        dbengines_unique_name: None,
        match_state_field: None,
        label: None
    }
    df_res = pd.DataFrame(columns=list(init_d.keys()))

    x_words_allin_y = lambda x, y: all(x_i in y for x_i in x) if len(x) <= len(y) else False

    matched_x = []
    matched_y = []
    single_x = []
    single_y = []
    assert(uriform_validate(df1[key_df1].values))
    assert(uriform_validate(df2[key_df2].values))
    for x_key_str in df1[key_df1]:
        temp_d = copy.deepcopy(init_d)
        temp_d[dbdbio_unique_name] = x_key_str
        x_key_words = x_key_str.split("-")
        y_key_str_matched = []
        for y_key_str in df2[key_df2]:
            y_key_words = y_key_str.split("-")
            if x_words_allin_y(x_key_words, y_key_words):
                y_key_str_matched.append(y_key_str)
        matched_y += y_key_str_matched
        if len(y_key_str_matched) > 1:
            temp_d[dbengines_unique_name] = ','.join(y_key_str_matched)
            temp_d[match_state_field] = "Multiple"
            print(f"MultiMatchWarning! x_words_allin_y pair: {x_key_str}, {y_key_str_matched}")
        elif len(y_key_str_matched) == 1:
            temp_d[dbengines_unique_name] = y_key_str_matched[0]
            if str(x_key_str).lower() == str(y_key_str_matched[0]).lower():
                temp_d[match_state_field] = "Normal"
                temp_d[label] = "Y_auto"  # Y is the abbreviation of YES
                # print(f"x_words_allin_y pair: {x_key_str}, {y_key_str_matched[0]}")
            else:
                temp_d[match_state_field] = "Fuzzy"
                print(f"FuzzyMatchWarning! x_words_allin_y pair: {x_key_str}, {y_key_str_matched[0]}")
        else:
            temp_d[dbengines_unique_name] = None
            temp_d[match_state_field] = "X_Single"
            temp_d[label] = "Y_auto"
            single_x.append(x_key_str)
        df_res = pd.concat([df_res, pd.DataFrame([temp_d])], axis=0, ignore_index=True)

    single_x = sorted(list(set(single_x)))
    matched_y = sorted(list(set(matched_y)))
    for x in df1[key_df1]:
        if x not in single_x:
            matched_x.append(x)
    for y in df2[key_df2]:
        temp_d = copy.deepcopy(init_d)
        if y not in matched_y:
            temp_d[merged_key_alias] = ""
            temp_d[dbdbio_unique_name] = ""
            temp_d[dbengines_unique_name] = y
            temp_d[match_state_field] = "Y_Single"
            temp_d[label] = "Y_auto"
            single_y.append(y)
            df_res = pd.concat([df_res, pd.DataFrame([temp_d])], axis=0, ignore_index=True)
    # print(f"UnmatchInfo! single_x: {single_x}")
    # print(f"UnmatchInfo! single_y: {single_y}")

    df_res[merged_key_alias] = df_res.apply(lambda df: unique_name_recalc(df[dbdbio_unique_name], df[dbengines_unique_name]), axis=1)
    if not len(df_res[merged_key_alias]) == len(set(df_res[merged_key_alias])):
        print(f"MultiValueWarning: Column {merged_key_alias} has duplicate values, please manually check the following values:")
        dup_key_index_dict = {k: tuple(d.index) for k, d in df_res.groupby(merged_key_alias) if len(d) > 1}
        print(f"\tdup_key_index_dict: {dup_key_index_dict}")
    df_res.to_csv(save_path, encoding=encoding, index=False)
    return


def unique_name_recalc(dbdbio_s, dbengines_s):
    s = dbdbio_s if pd.notna(dbdbio_s) else dbengines_s
    return key_uriform(s)


def merge_info_dbdbio_dbengines(df1, df2, df_feat_mapping_manulabeled, save_path, df_feature_mapping,
                                input_key_colname="key", key_avoid_conf_prefixes=("X_", "Y_"),
                                use_columns_merged=None, conflict_delimiter="#dbdbio>|<dbengines#", encoding="utf-8"):
    name_df1, name_df2, name_df_res = tuple(df_feature_mapping.index.values)
    df_res_key = df_feature_mapping[input_key_colname][name_df_res]
    full_columns_feature_mapping_dupkeys = list(df_feat_mapping_manulabeled.columns) + list(df_feature_mapping.loc[name_df_res].values)
    full_columns_feature_mapping = []
    for e in full_columns_feature_mapping_dupkeys:
        if e != df_res_key:
            full_columns_feature_mapping.append(e)
    full_columns_feature_mapping = [df_res_key] + full_columns_feature_mapping
    use_columns_merged = use_columns_merged or full_columns_feature_mapping
    assert(df_feature_mapping[input_key_colname][name_df_res] in use_columns_merged)
    use_columns_merged_mapping_part = [df_feature_mapping[input_key_colname][name_df_res]]
    for c in use_columns_merged:
        if c in df_feature_mapping.loc[name_df_res].values and c not in use_columns_merged_mapping_part:
            use_columns_merged_mapping_part.append(c)
    use_columns_mapping_part_flags = [c in use_columns_merged for c in df_feature_mapping.loc[name_df_res].values]
    use_columns_mapping_table = df_feature_mapping.columns[use_columns_mapping_part_flags].values

    df_res = copy.deepcopy(df_feat_mapping_manulabeled)
    key_df1_prefixed = ""
    key_df2_prefixed = ""
    for j_col in range(len(use_columns_merged_mapping_part)):
        colname = use_columns_mapping_table[j_col]
        temp_colname_df1, temp_colname_df2, temp_colname_df_res = tuple(df_feature_mapping[colname].values)
        key_column_on_process = colname == input_key_colname
        if key_column_on_process:
            key_df1 = temp_colname_df1
            key_df2 = temp_colname_df2
            df1.set_index(key_df1, inplace=True)
            df2.set_index(key_df2, inplace=True)
            key_df1_prefixed = key_avoid_conf_prefixes[0] + key_df1
            key_df2_prefixed = key_avoid_conf_prefixes[1] + key_df2
        if pd.isna(temp_colname_df1) and pd.isna(temp_colname_df2):
            raise ValueError(f"At least one of the input columns is not None! Please check the settings in colnames_mapping.csv!")
        elif pd.notna(temp_colname_df1) and pd.isna(temp_colname_df2):
            temp_values = []
            temp_df1_series = df_feat_mapping_manulabeled[key_df1_prefixed]
            for k_rec in range(len(df_feat_mapping_manulabeled)):
                try:
                    if key_column_on_process:
                        temp_item_df1 = temp_df1_series[k_rec]
                    else:
                        temp_rec_df1 = df1.loc[temp_df1_series[k_rec]]
                        temp_item_df1 = temp_rec_df1[temp_colname_df1]
                except KeyError:
                    temp_item_df1 = np.nan
                temp_values.append(temp_item_df1)
            df_res[temp_colname_df_res] = temp_values
        elif pd.isna(temp_colname_df1) and pd.notna(temp_colname_df2):
            temp_values = []
            temp_df2_series = df_feat_mapping_manulabeled[key_df2_prefixed]
            for k_rec in range(len(df_feat_mapping_manulabeled)):
                try:
                    if key_column_on_process:
                        temp_item_df2 = temp_df2_series[k_rec]
                    else:
                        temp_rec_df2 = df2.loc[temp_df2_series[k_rec]]
                        temp_item_df2 = temp_rec_df2[temp_colname_df2]
                except KeyError:
                    temp_item_df2 = np.nan
                temp_values.append(temp_item_df2)
            df_res[temp_colname_df_res] = temp_values
        else:
            # for i_loc in range(len(df_feature_mapping.columns)):
            temp_values = []
            temp_df1_series = df_feat_mapping_manulabeled[key_df1_prefixed]
            temp_df2_series = df_feat_mapping_manulabeled[key_df2_prefixed]
            for k_rec in range(len(df_feat_mapping_manulabeled)):
                try:
                    if key_column_on_process:
                        temp_item_df1 = temp_df1_series[k_rec]
                    else:
                        temp_rec_df1_index = temp_df1_series[k_rec]
                        temp_rec_df1 = df1.loc[temp_rec_df1_index]
                        temp_item_df1 = temp_rec_df1[temp_colname_df1]
                except KeyError:
                    temp_item_df1 = np.nan
                try:
                    if key_column_on_process:
                        temp_item_df2 = temp_df2_series[k_rec]
                    else:
                        temp_rec_df2_index = temp_df2_series[k_rec]
                        temp_rec_df2 = df2.loc[temp_rec_df2_index]
                        temp_item_df2 = temp_rec_df2[temp_colname_df2]
                except KeyError:
                    temp_item_df2 = np.nan
                if pd.notna(temp_item_df1) and pd.notna(temp_item_df2):
                    if temp_item_df1 == temp_item_df2:
                        v = temp_item_df1
                    else:
                        v = str(temp_item_df1) + conflict_delimiter + str(temp_item_df2)
                else:
                    v = temp_item_df1 if pd.notna(temp_item_df1) else temp_item_df2
                temp_values.append(v)
            df_res[temp_colname_df_res] = temp_values
    df_res = df_res[use_columns_merged]
    df_res.to_csv(save_path, encoding=encoding, index=False)
    print(save_path, 'saved!')
    return


if __name__ == '__main__':
    encoding = "utf-8"
    Base_Dir = pkg_rootdir
    month_yyyyMM = "202302"
    format_time_in_filename = "%Y%m"
    format_time_in_colname = "%b-%Y"
    curr_month = TimeFormat(month_yyyyMM, format_time_in_filename, format_time_in_filename)
    src_indiv_preprocessing_dir = os.path.join(Base_Dir, "data/db_indiv_preprocessing")
    tar_dbfeatfusion_dir = os.path.join(Base_Dir, "data/db_feature_fusion")
    # Step1: name alignment
    src_dbdbio_info_path = os.path.join(src_indiv_preprocessing_dir, f"OSDB_info_{month_yyyyMM}_joined_preprocessed.csv")
    src_dbengines_info_path = os.path.join(src_indiv_preprocessing_dir, f"ranking_crawling_{month_yyyyMM}_automerged_preprocessed.csv")
    tar_dbfeatfusion_dbname_mapping_autogen_path = os.path.join(tar_dbfeatfusion_dir, f"dbfeatfusion_dbname_mapping_{month_yyyyMM}_autogen.csv")

    dbdbio_info_dtype = {'Start Year': str, 'End Year': str}
    dbengines_info_dtype = {'initial_release_recalc': str, 'current_release_recalc': str, f'Rank_{curr_month.get_curr_month(format_time_in_colname)}': str}

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

    # Step2: DBMS features fusion
    settings_colnames_mapping_path = os.path.join(Base_Dir, "data/mapping_table/colnames_mapping.csv")
    df_settings_colnames_mapping = pd.read_csv(settings_colnames_mapping_path, encoding=encoding, index_col="tables")

    tar_dbfeatfusion_path = os.path.join(tar_dbfeatfusion_dir, f"dbfeatfusion_records_{month_yyyyMM}_automerged.csv")

    df_dbfeatfusion_dbname_mapping_manulabeled = pd.read_csv(src_dbfeatfusion_dbname_mapping_manulabeled_path, encoding=encoding, index_col=False)
    dbdbio_manulabed_flag_series = df_dbdbio_info_platform_filtered[key_dbdbio_info].apply(lambda x: x in df_dbfeatfusion_dbname_mapping_manulabeled[key_dbdbio_prefixed].values)
    dbengines_manulabed_flag_series = df_dbengines_info_platform_filtered[key_dbengines_info].apply(lambda x: x in df_dbfeatfusion_dbname_mapping_manulabeled[key_dbengines_prefixed].values)
    df_dbdbio_info_platform_filtered_manulabed = df_dbdbio_info_platform_filtered[dbdbio_manulabed_flag_series.values]
    df_dbengines_info_platform_filtered_manulabed = df_dbengines_info_platform_filtered[dbengines_manulabed_flag_series]

    # use_columns_merged = None
    use_columns_merged = ["DBMS_uriform", "X_DBMS_uriform", "Y_DBMS_uriform",
                          # "match_state", "manu_labeled_flag",
                          "DBMS_common_name",
                          "category_label", "Written in", "Query Interface", "System Architecture", "Developer",
                          # "Country of Origin", "initial_release_year", "current_release_year", "Project Type",
                          "License_info", "github_repo_link", "open_source_license", "Score_Feb-2023", "Rank_Feb-2023"]
    merge_info_dbdbio_dbengines(df_dbdbio_info_platform_filtered_manulabed, df_dbengines_info_platform_filtered_manulabed,
                                df_dbfeatfusion_dbname_mapping_manulabeled, save_path=tar_dbfeatfusion_path,
                                df_feature_mapping=df_settings_colnames_mapping, input_key_colname="key",
                                use_columns_merged=use_columns_merged, encoding=encoding)
