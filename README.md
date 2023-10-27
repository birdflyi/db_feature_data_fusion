# db_feature_data_fusion
Database feature data fusion for repositories [birdflyi/db_engines_ranking_table_crawling](https://github.com/birdflyi/db_engines_ranking_table_crawling) and [birdflyi/dbdbio_OSDB_info_crawling](https://github.com/birdflyi/dbdbio_OSDB_info_crawling).

## 1. Data processing workflow

### Step1: preprocessing
Preprocess data from [birdflyi/db_engines_ranking_table_crawling](https://github.com/birdflyi/db_engines_ranking_table_crawling) and [birdflyi/dbdbio_OSDB_info_crawling](https://github.com/birdflyi/dbdbio_OSDB_info_crawling), 
and save them into directory [db_feature_fusion](./data/db_feature_fusion).
The main task is to preprocess the fields corresponding to the key in [colnames_mapping.csv](./data/mapping_table/colnames_mapping.csv).

Try: 
- set month_yyyyMM = "yyyyMM"; 
- set curr_stage = 0; 
- run main.py.

### Step2: name alignment
Filter github open source projects, and merge the 'key' of dbdbio and dbengines according to the key in [colnames_mapping.csv](./data/mapping_table/colnames_mapping.csv).
The important columns in this step are: 'match_state_field' and 'manu_labeled_flag'.
'match_state_field' has 3 final states and 2 abnormal states: 
- 3 final states : ["Normal", "X_Single", "Y_Single"]
- 2 abnormal states: ["Fuzzy", "Multiple"]

Fuzzy Match means all words in dbdbio key contained by dbengines key.
Multiple Match means a dbdbio key can Fuzzy Match multiple dbengines keys.
Use the separator ":" to separate the auto match state and manu labeled match state.

'manu_labeled_flag' has default value empty string "" and 3 other values:
- "Y_auto" for automatched final states
- "Y" for "Fuzzy:(final state)" match state.
- "Y_ConflictResolved" for "Multiple:(final state)" match state.

Try: 
- manu-label dbfeatfusion_dbname_mapping_{month_yyyyMM}_manulabeled.csv based on the last month version.

### Step3: DBMS features fusion
Merge dbdbio and dbengines data tables in [db_feature_fusion](./data/db_feature_fusion) according to dbname mapping table(e.g. [dbfeatfusion_dbname_mapping_202302_manulabeled.csv](./data/mapping_table/dbfeatfusion_dbname_mapping_202302_manulabeled.csv)). 
Columns mapping use [colnames_mapping.csv](./data/mapping_table/colnames_mapping.csv).
Save the result table to dbfeatfusion records table(e.g. [dbfeatfusion_records_202302_automerged.csv](./data/db_feature_fusion/dbfeatfusion_records_202302_automerged.csv)).
The default separator setting when values conflicts during fusion: `conflict_delimiter="#dbdbio>|<dbengines#"`

Try:
- set curr_stage = 1;
- run main.py

### Step4: Solve conflicts manually
Conflicts occurs in each item contains conflict_delimiter(default "#dbdbio>|<dbengines#").
Solve conflicts manually as table "dbfeatfusion_records_{month_yyyyMM}_manulabeled.csv"(e.g. [dbfeatfusion_records_202302_manulabeled.csv](./data/db_feature_fusion/dbfeatfusion_records_202302_manulabeled.csv)).

Try:
- manu-label dbfeatfusion_records_{month_yyyyMM}_automerged_manulabeled_main_part.csv; 
- set curr_stage = 2; 
- run main.py.

## 2. How to update data

### Step1: Update git submodules
Use git command in the root directory of this data fusion project to update each git submodule:
```git
git submodule foreach git checkout main
git submodule foreach git pull
```

### Step2: Make changes and push
Follow the instructions in step "1. Data processing workflow" to make changes. Push the commits to origin.
