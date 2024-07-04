#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
import sys
import shutil
from pathlib import Path
import argparse
from fets_challenge import run_challenge_experiment
from fets_challenge.experiment import logger

def main(argv, trg_folder, trg_path, brats_training_data_parent_dir):
    # subset split by mean of poisson distribution on every nodes
    df = pd.read_csv(trg_path)
    original_unique_IDs = df['Partition_ID'].unique()
    # print(f"original_unique_IDs({len(original_unique_IDs)}): {original_unique_IDs}")
    frequency = df['Partition_ID'].value_counts()
    lambda_ = frequency.mean()
    max_size = frequency.max()
    std_ = np.sqrt(lambda_)
    margins = lambda_ + (argv.z_score * std_)
    # lower_bound = lambda_ - (argv.z_score * std_)
    subset_size = int(max(1, margins))

    df['Partition_ID'] = df['Partition_ID'].astype(int) * 100
    df['Partition_ID'] = df['Partition_ID'].astype(str)
    original_x100_unique_IDs = df['Partition_ID'].unique()
    # print(f"original_x100_unique_IDs({len(original_x100_unique_IDs)}): {original_x100_unique_IDs}")

    for pid in original_x100_unique_IDs:
        indices = df[df['Partition_ID'] == pid].index
        df.loc[indices, 'Partition_ID'] = [str(int(pid) + i // subset_size) for i in range(len(indices))]

    subset_x100_unique_IDs = df['Partition_ID'].unique()
    # print(f"subset_x100_unique_IDs({len(subset_x100_unique_IDs)}): {subset_x100_unique_IDs}")

    # node_ids = np.unique(np.array([int(el)//100 for el in unique_values])).tolist()
    primary_group = [int(el) for el in original_unique_IDs if frequency.loc[el] > lambda_]
    # print(f"primary_group({len(primary_group)}): {primary_group}")

    secondary_group = [int(el) for el in original_unique_IDs if el not in primary_group]
    # print(f"secondary_group({len(secondary_group)}): {secondary_group}")

    # print(f'subset_group: {subset_group}')
    # subset_dict = {}
    # for original_ID in original_unique_IDs:
    #     subset_dict[original_ID] = [el for el in subset_group if int(original_ID) == int(el)//100]
    subset_dict = {k:[el for el in subset_x100_unique_IDs if int(k) == int(el)//100] for k in original_unique_IDs}
    # print(subset_dict)
    primary_subset_dict = {k:v for k,v in subset_dict.items() if k in primary_group}
    for orig,subs in primary_subset_dict.items():
        print(f"PRIMARY[{orig}]({len(subs)})({subset_size}): {subs}")
    # print(primary_subset_dict)
    secondary_subset_dict = {k:v for k,v in subset_dict.items() if k in secondary_group}
    for orig,subs in secondary_subset_dict.items():
        print(f"SECONDARY[{orig}]({len(subs)})({subset_size}): {subs}")

    out_path = os.path.join(trg_folder, f"subset_{argv.institution_split_csv_filename}")    
    df.to_csv(out_path, index=False)
    # print(f"Data saved to subset_{argv.institution_split_csv_filename}.")
    institution_split_csv_filename = f"subset_{argv.institution_split_csv_filename}"


    def FedPOD_collaborators_on_rounds(collaborators,
                                        db_iterator,
                                        fl_round,
                                        collaborators_chosen_each_round,
                                        collaborator_times_per_round):
        # csv_file = argv.institution_split_csv_filename
        
        if fl_round >= 0:
            n_nodes = 8
        if fl_round >= 5:
            n_nodes = 10
        if fl_round >= 10:
            n_nodes = 15
        if fl_round >= 15:
            n_nodes = 20
        # n_nodes = argv.n_nodes

        # node_ids = np.unique(np.array([int(el)//100 for el in collaborators])).tolist()
        # major_group = [1, 18] if len(node_ids) == 23 else [1, 2, 3, 24, 25, 26]
        # minor_group = [el for el in node_ids if el not in major_group]
                                            
        major_np = np.array(primary_group)
        minor_np = np.array(secondary_group)
        np.random.shuffle(major_np)
        np.random.shuffle(minor_np)
        major_list = major_np.tolist()
        minor_list = minor_np.tolist()
            
        n_major = len(major_list) if n_nodes >= len(major_list) else n_nodes
        n_minor = max(0, n_nodes-len(major_list))
        nodes_selected = [
            *major_list[:n_major],
            *minor_list[:n_minor]
        ]
                                            
        # subset_list = [int(el) for el in collaborators]
        # subset_np = np.array(subset_list)/100

        # subsets_selected = []
        # for node_id in nodes_selected:
        #     subsets_of_node = subset_np[subset_np.astype('int') == node_id] * 100
        #     np.random.shuffle(subsets_of_node) 
        #     subsets_selected = [*subsets_selected, int(subsets_of_node[0])]
            
        subsets_selected = []
        for node_id in nodes_selected:
            subsets_of_node = subset_dict[node_id]
            np.random.shuffle(subsets_of_node) 
            subsets_selected = [*subsets_selected, int(subsets_of_node[0])]

        return [str(el) for el in subsets_selected]


    def FedPOD_parameters(collaborators,
                                    db_iterator,
                                    fl_round,
                                    collaborators_chosen_each_round,
                                    collaborator_times_per_round):
        if fl_round >= 0:
            epochs_per_round = 4
        if fl_round >= 5:
            epochs_per_round = 3
        if fl_round >= 10:
            epochs_per_round = 3
        if fl_round >= 15:
            epochs_per_round = 2

        learning_rate = 1e-3
        
        return (learning_rate, epochs_per_round)


    def FedPOD_aggregation(local_tensors,
                                    tensor_db,
                                    tensor_name,
                                    fl_round,
                                    collaborators_chosen_each_round,
                                    collaborator_times_per_round):
        if fl_round == 0:
            weight = [t.weight for t in local_tensors]
            tensor_values = [t.tensor for t in local_tensors]
            return np.average(tensor_values, weights=weight, axis=0)
        else:
            col_names = [t.col_name for t in local_tensors]
            # 검색 조건 설정
            pre_tags = [(el, 'metric', 'validate_agg') for el in collaborators_chosen_each_round[fl_round]]
            post_tags = [(el, 'metric', 'validate_local') for el in collaborators_chosen_each_round[fl_round]]
            # search_tags = [('1', 'metric'), ('2', 'metric'), ('3', 'metric')]

            # 조건에 맞는 데이터 필터링
            pre_df = tensor_db[
                (tensor_db['tensor_name'] == 'valid_loss') &
                (tensor_db['round'] == (fl_round)) &
                (tensor_db['tags'].apply(lambda x: x in pre_tags)) &
                (tensor_db['origin'] == 'aggregator')
            ]
            pre_loss_dict = {row['tags'][0]: float(row['nparray']) for index, row in pre_df.iterrows()}
            post_df = tensor_db[
                (tensor_db['tensor_name'] == 'valid_loss') &
                (tensor_db['round'] == (fl_round)) &
                (tensor_db['tags'].apply(lambda x: x in post_tags)) &
                (tensor_db['origin'] == 'aggregator')
            ]
            post_loss_dict = {row['tags'][0]: float(row['nparray']) for index, row in post_df.iterrows()}


            pre_cost = [pre_loss_dict[col_name] for col_name in col_names]
            post_cost = [post_loss_dict[col_name] for col_name in col_names]
            deriv = [max(0, pre - post) for (pre, post) in zip(pre_cost, post_cost)]
            total_deriv = sum(deriv) + 1e-10
            deriv = [el/total_deriv for el in deriv]

            integ = [min(pre, post) + (max(0, pre - post)/2) for (pre, post) in zip(pre_cost, post_cost)]
            total_integ = sum(integ)
            integ = [el / total_integ for el in integ]
            
            weight = [t.weight for t in local_tensors]
            
            VPID = [0.45*w+0.1*m+0.45*k for (w, m, k) in zip(weight, integ, deriv)]

            tensor_values = [t.tensor for t in local_tensors]
            return np.average(tensor_values, weights=VPID, axis=0)
            
    device = 'cuda'
    db_store_rounds = 1
    save_checkpoints = True
    
    # change any of these you wish to your custom functions. You may leave defaults if you wish.
    aggregation_function = FedPOD_aggregation # weighted_average_aggregation# FedAvgM_Selection  # weighted_average_aggregation
    choose_training_collaborators = FedPOD_collaborators_on_rounds # all_collaborators_train
    training_hyper_parameters_for_round = FedPOD_parameters # constant_hyper_parameters

    include_validation_with_hausdorff=False
    # the scores are returned in a Pandas dataframe
    scores_dataframe, checkpoint_folder = run_challenge_experiment(
        aggregation_function=aggregation_function,
        choose_training_collaborators=choose_training_collaborators,
        training_hyper_parameters_for_round=training_hyper_parameters_for_round,
        include_validation_with_hausdorff=include_validation_with_hausdorff,
        institution_split_csv_filename=institution_split_csv_filename,
        brats_training_data_parent_dir=brats_training_data_parent_dir,
        db_store_rounds=db_store_rounds,
        rounds_to_train=argv.rounds_to_train,
        device=device,
        save_checkpoints=save_checkpoints,
        restore_from_checkpoint_folder = argv.restore_from_checkpoint_folder)
    scores_dataframe

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-C', '--restore_from_checkpoint_folder', type=str, default=None)
    parser.add_argument('-W', '--workspace', type=str, default='workspace')
    # workspace = 'workspace'
    parser.add_argument('-R', '--rounds_to_train', type=int, required=True)
    # assert isinstance(int(sys.argv[2]), int), f"{sys.argv[2]} must be integer"
    # rounds_to_train = int(sys.argv[2])
    parser.add_argument('-F', '--institution_split_csv_filename', type=str, required=True)
    # institution_split_csv_filename = sys.argv[3]
    # parser.add_argument('-E', '--epochs_per_round', type=int, required=True)
    # assert isinstance(int(sys.argv[4]), int), f"{sys.argv[4]} must be integer"
    # argv.epochs_per_round = int(sys.argv[4])
    # parser.add_argument('-M', '--milestone', type=int, required=True)
    # assert isinstance(int(sys.argv[5]), int), f"{sys.argv[5]} must be integer"
    # _milestone = int(sys.argv[5])
    # parser.add_argument('-N', '--n_nodes', type=int, required=True)
    # assert isinstance(int(sys.argv[6]), int), f"{sys.argv[6]} must be integer"
    # argv.n_nodes = int(sys.argv[6]) # sys.argv[?]
    parser.add_argument('-Z', '--z_score', type=float, required=True)
    # assert isinstance(float(sys.argv[7]), float), f"{sys.argv[7]} must be float"
    # argv.z_score = float(sys.argv[7]) # sys.argv[?]
    argv = parser.parse_args(sys.argv[1:])
    print(argv)

    home = str(Path.home())
    trg_folder = os.path.join(home, f'.local', argv.workspace)
    trg_path = os.path.join(trg_folder, argv.institution_split_csv_filename)

    # assert os.path.exists(trg_path), f"{trg_path} not exists"
    # ckp_path = os.path.join(trg_folder, 'checkpoint', argv.restore_from_checkpoint_folder)
    # if argv.restore_from_checkpoint_folder is None: 
    #     print('Initial Mode')
    # else: 
    #     assert os.path.exists(ckp_path), f"{ckp_path} not exists" 
    brats_training_data_parent_dir = os.path.join(home, '2024_data', 'FeTS2022', 'center')# f'/home2/{os.getlogin()}/2024_data/FeTS2022/center'
    # assert os.path.isdir(brats_training_data_parent_dir), f"not exist folder {brats_training_data_parent_dir}"
    args = [
        trg_folder,
        trg_path,
        brats_training_data_parent_dir
    ]
    main(argv, *args)