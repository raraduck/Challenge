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
    frequency = df['Partition_ID'].value_counts()
    lambda_ = frequency.mean()
    std_ = np.sqrt(lambda_)
    margins = lambda_ + (argv.z_score * std_)
    # upper_bound = lambda_ + (argv.z_score * std_)
    subset_size = int(max(1, margins))

    df['Partition_ID'] = df['Partition_ID'].astype(int) * 100
    df['Partition_ID'] = df['Partition_ID'].astype(str)
    original_x100_unique_IDs = df['Partition_ID'].unique()

    for pid in original_x100_unique_IDs:
        indices = df[df['Partition_ID'] == pid].index
        df.loc[indices, 'Partition_ID'] = [str(int(pid) + i // subset_size) for i in range(len(indices))]

    subset_x100_unique_IDs = df['Partition_ID'].unique()

    primary_group = [int(el) for el in original_unique_IDs if frequency.loc[el] > lambda_ + (1.96 * std_)]

    secondary_group = [int(el) for el in original_unique_IDs if el not in primary_group]
    subset_dict = {k:[el for el in subset_x100_unique_IDs if int(k) == int(el)//100] for k in original_unique_IDs}
    primary_subset_dict = {k:v for k,v in subset_dict.items() if k in primary_group}
    for orig,subs in primary_subset_dict.items():
        print(f"PRIMARY[{orig}]({len(subs)})({subset_size}): {subs}")
    secondary_subset_dict = {k:v for k,v in subset_dict.items() if k in secondary_group}
    for orig,subs in secondary_subset_dict.items():
        print(f"SECONDARY[{orig}]({len(subs)})({subset_size}): {subs}")

    out_path = os.path.join(trg_folder, f"subset_{argv.institution_split_csv_filename}")    
    df.to_csv(out_path, index=False)
    institution_split_csv_filename = f"subset_{argv.institution_split_csv_filename}"


    def FedPOD_collaborators_on_rounds(collaborators,
                                        db_iterator,
                                        fl_round,
                                        collaborators_chosen_each_round,
                                        collaborator_times_per_round):
        # csv_file = argv.institution_split_csv_filename
        if argv.institution_split_csv_filename == 'partitioning_0.csv':
            if fl_round >= 0:
                n_nodes = 4
            if fl_round >= 5:
                n_nodes = 5
            if fl_round >= 10:
                n_nodes = 10
            if fl_round >= 15:
                n_nodes = 12
        elif argv.institution_split_csv_filename == 'partitioning_1.csv':
            if fl_round >= 0:
                n_nodes = 6
            if fl_round >= 5:
                n_nodes = 8
            if fl_round >= 10:
                n_nodes = 10
            if fl_round >= 15:
                n_nodes = 12
            if fl_round >= 20:
                n_nodes = 15
        elif argv.institution_split_csv_filename == 'partitioning_2.csv':
            if fl_round >= 0:
                n_nodes = 8
            if fl_round >= 5:
                n_nodes = 10
            if fl_round >= 10:
                n_nodes = 12
            if fl_round >= 15:
                n_nodes = 15
            if fl_round >= 20:
                n_nodes = 18
        elif argv.institution_split_csv_filename == 'partitioning_3.csv':
            if fl_round >= 0:
                n_nodes = 6
            if fl_round >= 5:
                n_nodes = 8
            if fl_round >= 10:
                n_nodes = 10
            if fl_round >= 15:
                n_nodes = 15
            if fl_round >= 20:
                n_nodes = 20
        else:
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
        if argv.institution_split_csv_filename == 'partitioning_1.csv':
            major_list = [*major_list, *major_list, *major_list]
        elif argv.institution_split_csv_filename == 'partitioning_2.csv':
            major_list = [*major_list, *major_list]
        else:
            major_list = [*major_list, *major_list]
        minor_list = minor_np.tolist()
            
        n_major = len(major_list) if n_nodes > len(major_list) else n_nodes
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
            for subset_id in subsets_of_node:
                if int(subset_id) not in subsets_selected:
                    subsets_selected = [*subsets_selected, int(subset_id)]
                    break
            # subsets_selected = [*subsets_selected, int(subsets_of_node[0])]

        return [str(el) for el in subsets_selected]


    def FedPOD_parameters(collaborators,
                                    db_iterator,
                                    fl_round,
                                    collaborators_chosen_each_round,
                                    collaborator_times_per_round):
        if argv.institution_split_csv_filename == 'partitioning_0.csv':
            if fl_round >= 0:
                epochs_per_round = 4
                learning_rate = 5e-3
            if fl_round >= 5:
                epochs_per_round = 4
                learning_rate = 2e-3
            if fl_round >= 10:
                epochs_per_round = 3
                learning_rate = 1e-3
            if fl_round >= 15:
                epochs_per_round = 3
                learning_rate = 1e-3
        elif argv.institution_split_csv_filename == 'partitioning_1.csv':
            if fl_round >= 0:
                epochs_per_round = 4
            if fl_round >= 5:
                epochs_per_round = 4
            if fl_round >= 10:
                epochs_per_round = 3
            if fl_round >= 15:
                epochs_per_round = 3
            learning_rate = 1e-3
        elif argv.institution_split_csv_filename == 'partitioning_2.csv':
            if fl_round >= 0:
                epochs_per_round = 4
            if fl_round >= 5:
                epochs_per_round = 4
            if fl_round >= 10:
                epochs_per_round = 3
            if fl_round >= 15:
                epochs_per_round = 3
            learning_rate = 1e-3
        elif argv.institution_split_csv_filename == 'partitioning_3.csv':
            if fl_round >= 0:
                epochs_per_round = 4
                learning_rate = 5e-3
            if fl_round >= 5:
                epochs_per_round = 4
                learning_rate = 2e-3
            if fl_round >= 10:
                epochs_per_round = 3
                learning_rate = 1e-3
            if fl_round >= 15:
                epochs_per_round = 3
                learning_rate = 1e-3
        else:
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
            weight = [t.weight for t in local_tensors]

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
            
            switch = [1 if pre > post else 0 for (pre, post) in zip(pre_cost, post_cost)]
            deriv = [max(0, pre - post) for (pre, post) in zip(pre_cost, post_cost)]

            deriv = [w*k for (w, k) in zip(weight, deriv)]
            total_deriv = sum(deriv) + 1e-10
            deriv = [el/total_deriv for el in deriv]

            # deriv = np.average(deriv, weights=weight, axis=0)

            integ = [min(pre, post) + (max(0, pre - post)/2) for (pre, post) in zip(pre_cost, post_cost)]

            integ = [w*m for (w, m) in zip(weight, integ)]
            total_integ = sum(integ)
            integ = [el / total_integ for el in integ]
            
            # integ = np.average(integ, weights=weight, axis=0)
            
            VPID = [0.2*w+0.1*m+0.7*k for (w, m, k) in zip(weight, integ, deriv)]
            # total_VPID = sum(VPID) + 1e-10
            # VPID = [el/total_VPID for el in VPID]

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

    if False:
        from fets_challenge import model_outputs_to_disc
        from pathlib import Path

        # infer participant home folder
        home = str(Path.home())

        # you will need to specify the correct experiment folder and the parent directory for
        # the data you want to run inference over (assumed to be the experiment that just completed)

        #checkpoint_folder='experiment_1'
        #data_path = </PATH/TO/CHALLENGE_VALIDATION_DATA>
        data_path = '/home/brats/MICCAI_FeTS2022_ValidationData'
        validation_csv_filename = 'validation.csv'

        # you can keep these the same if you wish
        final_model_path = os.path.join(home, '.local/workspace/checkpoint', checkpoint_folder, 'best_model.pkl')

        # If the experiment is only run for a single round, use the temp model instead
        if not Path(final_model_path).exists():
          final_model_path = os.path.join(home, '.local/workspace/checkpoint', checkpoint_folder, 'temp_model.pkl')

        outputs_path = os.path.join(home, '.local/workspace/checkpoint', checkpoint_folder, 'model_outputs')

        model_outputs_to_disc(data_path=data_path, 
                              validation_csv=validation_csv_filename,
                              output_path=outputs_path, 
                              native_model_path=final_model_path,
                              outputtag='',
                              device=device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-C', '--restore_from_checkpoint_folder', type=str, default=None)
    parser.add_argument('-W', '--workspace', type=str, default='workspace')
    parser.add_argument('-R', '--rounds_to_train', type=int, default=30)
    parser.add_argument('-F', '--institution_split_csv_filename', type=str, default='partitioning_2.csv')
    parser.add_argument('-Z', '--z_score', type=float, default=-1.75)
    argv = parser.parse_args(sys.argv[1:])
    print(argv)

    home = str(Path.home())
    trg_folder = os.path.join(home, f'.local', argv.workspace)
    trg_path = os.path.join(trg_folder, argv.institution_split_csv_filename)

    # brats_training_data_parent_dir = os.path.join(home, '2024_data', 'FeTS2022', 'center')# f'/home2/{os.getlogin()}/2024_data/FeTS2022/center'
    brats_training_data_parent_dir = '/raid/datasets/FeTS22/MICCAI_FeTS2022_TrainingData'
    args = [
        trg_folder,
        trg_path,
        brats_training_data_parent_dir
    ]
    main(argv, *args)
