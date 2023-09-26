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


def key_urnform(key_common_name_str):
    key_common_name_str = str(key_common_name_str).lower().replace('.', '')  # remove '.'
    key_common_name_splits = re.split("[- /\():,]", key_common_name_str.lower())
    key_common_name_splits_striped = [s.strip() for s in key_common_name_splits if s.strip() != '']  # deal with the case: "Apache Jena - TDB"
    key_urnform = '-'.join(key_common_name_splits_striped)
    return key_urnform


def dbdbio_feat_preprocessing(src_path, tar_path, **kwargs):
    dtype = kwargs.get("dtype")
    df_dbdbio_info = pd.read_csv(src_path, encoding=encoding, index_col=False, dtype=dtype)
    # preprocessings
    # map labels
    label_col = kwargs.get("label_col", "Data_Model_mapping")
    df_dbdbio_info[label_col] = df_dbdbio_info[label_col].apply(mapping_values2labels, **validate_label_mapping_table(df_dbdbio_info[label_col]))

    # DBMS_urnform_recalc = df_dbdbio_info["card_title"].apply(key_urnform)
    # for i in range(len(DBMS_urnform_recalc)):
    #     if DBMS_urnform_recalc.values[i] != df_dbdbio_info["DBMS_urnform"].values[i]:
    #         print(f"Warning: Auto generated URN{DBMS_urnform_recalc.values[i]} is not matched with {df_dbdbio_info['DBMS_urnform'].values[i]}")
    df_dbdbio_info.to_csv(tar_path, encoding=encoding, index=False)
    return


def dbengines_feat_preprocessing(src_path, tar_path, index_col=False, **kwargs):
    dtype = kwargs.get("dtype", str)
    df_dbengines_info = pd.read_csv(src_path, encoding=encoding, index_col=index_col, dtype=dtype)

    # preprocessings
    df_dbengines_info["DBMS_urnform"] = df_dbengines_info["DBMS"].apply(key_urnform)
    # map labels
    label_col = kwargs.get("label_col", "category_label")
    df_dbengines_info[label_col] = df_dbengines_info[label_col].apply(mapping_values2labels, **validate_label_mapping_table(df_dbengines_info[label_col]))

    df_dbengines_info.to_csv(tar_path, encoding=encoding, index=False)
    return


def validate_label_mapping_table(str_series, k_v_colnames=None, mapping_table_path=None, encoding="utf-8", index_col=False):
    elem_splited_notna = [[e.strip() for e in s.split(',')] for s in pd.Series(str_series).dropna()]
    elem_splited_flatten = sum(elem_splited_notna, [])  # use sum as the iterate tool
    elem_set_sorted = list(set(elem_splited_flatten))
    mapping_table_path = mapping_table_path or os.path.join(pkg_rootdir, f'data/existing_tagging_info/category_labels_mapping_table.csv')
    df_category_labels_mapping_table = pd.read_csv(mapping_table_path, encoding=encoding, index_col=index_col)
    k_v_colnames = k_v_colnames or ["category_label", "category_name"]
    category_name_col = df_category_labels_mapping_table[k_v_colnames[1]]
    # validate
    for e in elem_set_sorted:
        if e not in list(category_name_col):
            raise KeyError(f"The key '{e}' must be in category_name_col: {list(category_name_col)}. "
                           f"Check the category_name column in {mapping_table_path}!")
    raw_df_k_v_cols = df_category_labels_mapping_table[k_v_colnames]

    def merge2dict_df_k_v_cols(df, k_colname, v_colname):
        temp_dict = {}
        for i in range(len(df)):
            k = df.loc[i, k_colname]
            v = df.loc[i, v_colname]
            if temp_dict.get(k, None) is not None:
                temp_elem_list = temp_dict[k].split(',')
                temp_elem_list.append(v)
                temp_dict[k] = ','.join(temp_elem_list)
            else:
                temp_dict[k] = v
        return temp_dict

    dict_k_category_labels__v_category_names = merge2dict_df_k_v_cols(raw_df_k_v_cols, k_v_colnames[0], k_v_colnames[1])
    dict_k_category_names__v_category_labels = merge2dict_df_k_v_cols(raw_df_k_v_cols, k_v_colnames[1], k_v_colnames[0])
    mapping_dicts = {
        "raw_df_k_v_cols": raw_df_k_v_cols,
        "label_dict": dict_k_category_labels__v_category_names,
        "mapping_dict": dict_k_category_names__v_category_labels,
    }
    return mapping_dicts


def mapping_values2labels(item, **kwargs):
    mapping_dict = kwargs.get("mapping_dict")
    if not mapping_dict:
        raise KeyError("Key 'mapping_dict' can not be found!")
    if pd.isna(item):
        return item
    else:
        temp_item_list = [mapping_dict[e.strip()] for e in item.split(',')]  # e.g. "Object-Relational, Network"
        flatten_item_list = []
        for elem in temp_item_list:
            elem_list = elem.split(',')  # "Object oriented,Relational",Object-Relational: the key may be multi-types.
            flatten_item_list.append(elem_list)
        flatten_item_list = sum(flatten_item_list, [])
        flatten_item_list_dedup = list(set(flatten_item_list))
        flatten_item_list_dedup.sort(key=flatten_item_list.index)  # recover the order by the first hit index
        return ",".join(flatten_item_list_dedup)


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

    src_dbdbio_info_raw_path = os.path.join(src_dbdbio_dir, f"OSDB_info_{month_yyyyMM}_joined_manulabeled.csv")
    src_dbengines_info_raw_path = os.path.join(src_dbengines_dir, f"ranking_crawling_{month_yyyyMM}_automerged_manulabeled.csv")
    src_dbdbio_info_path = os.path.join(src_indiv_preprocessing_dir,
                                        f"OSDB_info_{month_yyyyMM}_joined_preprocessed.csv")
    src_dbengines_info_path = os.path.join(src_indiv_preprocessing_dir,
                                           f"ranking_crawling_{month_yyyyMM}_automerged_preprocessed.csv")

    dbdbio_info_dtype = {'Start Year': str, 'End Year': str}
    dbengines_info_dtype = {'initial_release_recalc': str, 'current_release_recalc': str, f'Rank_{curr_month.get_curr_month(format_time_in_colname)}': str}

    dbdbio_feat_preprocessing(src_dbdbio_info_raw_path, src_dbdbio_info_path, dtype=dbdbio_info_dtype)
    dbengines_feat_preprocessing(src_dbengines_info_raw_path, src_dbengines_info_path, dtype=dbengines_info_dtype)
