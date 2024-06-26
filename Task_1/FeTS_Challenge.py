#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
import sys
import shutil
from pathlib import Path
from fets_challenge import run_challenge_experiment
from fets_challenge.experiment import logger

# home = str(Path.home())
# workspace = sys.argv[3].split('.')[0]
# srcdir = os.path.join(home, '.local', 'workspace')
# trgdir = os.path.join(home, '.local', workspace)
# shutil.copytree(srcdir, trgdir)
# # os.makedirs(chdir, exist_ok=True)
# assert os.path.exists(trgdir), f"chdir not exist"
# os.chdir(trgdir)
workspace = 'workspace'
brats_training_data_parent_dir = f'/home2/{os.getlogin()}/2024_data/FeTS2022/center'
assert os.path.isdir(brats_training_data_parent_dir), f"not exist folder {brats_training_data_parent_dir}"
device = 'cuda'
validation_csv_filename = 'validation.csv'

if sys.argv[1] in ['train', 'training']:
    assert isinstance(int(sys.argv[2]), int), f"{sys.argv[2]} must be integer"
    rounds_to_train = int(sys.argv[2])

    institution_split_csv_filename = sys.argv[3] # 'FeTS2_stage1_2.csv'
    home = str(Path.home())
    trg_folder = os.path.join(home, f'.local/{workspace}')
    trg_path = os.path.join(trg_folder, institution_split_csv_filename)
    assert os.path.exists(trg_path), f"{trg_path} not exists"

    assert isinstance(int(sys.argv[4]), int), f"{sys.argv[4]} must be integer"
    _epochs_per_round = int(sys.argv[4])

    assert isinstance(int(sys.argv[5]), int), f"{sys.argv[5]} must be integer"
    _milestone = int(sys.argv[5])

    assert isinstance(int(sys.argv[6]), int), f"{sys.argv[6]} must be integer"
    _n_nodes = int(sys.argv[6]) # sys.argv[?]
    
    assert isinstance(float(sys.argv[7]), float), f"{sys.argv[7]} must be float"
    _z_score = float(sys.argv[7]) # sys.argv[?]

    if sys.argv[1] == 'training': 
        restore_from_checkpoint_folder = sys.argv[8] # "experiment_vj02_1"
        ckp_path = os.path.join(trg_folder, 'checkpoint', restore_from_checkpoint_folder)
        assert os.path.exists(ckp_path), f"{ckp_path} not exists"
    else:
        restore_from_checkpoint_folder = None

    db_store_rounds = 1
    save_checkpoints = True

    # subset split by mean of poisson distribution on every nodes
    df = pd.read_csv(trg_path)
    df['Partition_ID'] = df['Partition_ID'].astype(int) * 100
    df['Partition_ID'] = df['Partition_ID'].astype(str)

    unique_values = df['Partition_ID'].unique()

    frequency = df['Partition_ID'].value_counts()
    lambda_ = frequency.mean()
    max_size = frequency.max()
    # subset_factor = lambda_/max_size
    # subset_size = int(lambda_*subset_factor)
    std_ = np.sqrt(lambda_)
    upper_bound = lambda_ + (_z_score * std_)
    lower_bound = lambda_ - (_z_score * std_)
    subset_size = int(max(1, upper_bound))

    print(unique_values)
    print(df.columns)

    for pid in unique_values:
        indices = df[df['Partition_ID'] == pid].index
        df.loc[indices, 'Partition_ID'] = [str(int(pid) + i // subset_size) for i in range(len(indices))]

    out_path = os.path.join(trg_folder, f"subset_{institution_split_csv_filename}")    
    df.to_csv(out_path, index=False)
    print("Data saved to 'subset_partitioning.csv'.")

    institution_split_csv_filename = f"subset_{institution_split_csv_filename}"

    # exit(0)
    
    def subset_selection_for_collaborators_on_rounds(collaborators,
                                       db_iterator,
                                       fl_round,
                                       collaborators_chosen_each_round,
                                       collaborator_times_per_round):
        csv_file = institution_split_csv_filename
        n_nodes = _n_nodes
    
        node_ids = np.unique(np.array([int(el)//100 for el in collaborators])).tolist()
        major_group = [1, 18] if len(node_ids) == 23 else [1, 2, 3, 24, 25, 26]
        minor_group = [el for el in node_ids if el not in major_group]
                                           
        major_np = np.array(major_group)
        minor_np = np.array(minor_group)
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

        # if fl_round > 0:
        #     sorted_list = sorted(collaborator_times_per_round[fl_round-1].items(), key=lambda item: item[1])
        #     ordered_by_time = [int(el[0]) for el in sorted_list]
                                           
        subset_list = [int(el) for el in collaborators]
        subset_np = np.array(subset_list)/100
        subsets_selected = []
        for node_id in nodes_selected:
            subsets_of_node = subset_np[subset_np.astype('int') == node_id] * 100
            np.random.shuffle(subsets_of_node) 
            # if fl_round == 0:
            #     np.random.shuffle(subsets_of_node) 
            # else: 
            #     subsets_of_node = [el for el in ordered_by_time if el in subsets_of_node]
            subsets_selected = [*subsets_selected, int(subsets_of_node[0])]
    
        return [str(el) for el in subsets_selected]
                                       
    # a very simple function. Everyone trains every round.
    def all_collaborators_train(collaborators,
                                db_iterator,
                                fl_round,
                                collaborators_chosen_each_round,
                                collaborator_times_per_round):
        """Chooses which collaborators will train for a given round.
        
        Args:
            collaborators: list of strings of collaborator names
            db_iterator: iterator over history of all tensors.
                Columns: ['tensor_name', 'round', 'tags', 'nparray']
            fl_round: round number
            collaborators_chosen_each_round: a dictionary of {round: list of collaborators}. Each list indicates which collaborators trained in that given round.
            collaborator_times_per_round: a dictionary of {round: {collaborator: total_time_taken_in_round}}.  
        """
        return collaborators
    
    def major_minor_parameters(collaborators,
                                  db_iterator,
                                  fl_round,
                                  collaborators_chosen_each_round,
                                  collaborator_times_per_round):
        
        # major_epochs = _peoch
        # minor_epochs = 1
                                      
        epochs_per_round = _epochs_per_round # major_epochs if fl_round < milestone else minor_epochs
        milestone = _milestone
        learning_rate = 1e-3 if fl_round < milestone  else 1e-4
        
        return (learning_rate, epochs_per_round)

    
    def constant_hyper_parameters(collaborators,
                                  db_iterator,
                                  fl_round,
                                  collaborators_chosen_each_round,
                                  collaborator_times_per_round):
        """Set the training hyper-parameters for the round.
        
        Args:
            collaborators: list of strings of collaborator names
            db_iterator: iterator over history of all tensors.
                Columns: ['tensor_name', 'round', 'tags', 'nparray']
            fl_round: round number
            collaborators_chosen_each_round: a dictionary of {round: list of collaborators}. Each list indicates which collaborators trained in that given round.
            collaborator_times_per_round: a dictionary of {round: {collaborator: total_time_taken_in_round}}.  
        Returns:
            tuple of (learning_rate, epochs_per_round).
        """
        # these are the hyperparameters used in the May 2021 recent training of the actual FeTS Initiative
        # they were tuned using a set of data that UPenn had access to, not on the federation itself
        # they worked pretty well for us, but we think you can do better :)
        epochs_per_round = 1
        learning_rate = 1e-3
        return (learning_rate, epochs_per_round)
                                      
    
    # the simple example of weighted FedAVG
    def weighted_average_aggregation(local_tensors,
                                     tensor_db,
                                     tensor_name,
                                     fl_round,
                                     collaborators_chosen_each_round,
                                     collaborator_times_per_round):
        """Aggregate tensors. This aggregator clips all tensor values to the 80th percentile of the absolute values to prevent extreme changes.
    
        Args:
            local_tensors(list[openfl.utilities.LocalTensor]): List of local tensors to aggregate.
            tensor_db: pd.DataFrame that contains global tensors / metrics.
                Columns: ['tensor_name', 'origin', 'round', 'report',  'tags', 'nparray']
            tensor_name: name of the tensor
            fl_round: round number
            collaborators_chosen_each_round: a dictionary of {round: list of collaborators}. Each list indicates which collaborators trained in that given round.
            collaborator_times_per_round: a dictionary of {round: {collaborator: total_time_taken_in_round}}.
        """
        # basic weighted fedavg
    
        # here are the tensor values themselves
        tensor_values = [t.tensor for t in local_tensors]
        
        # and the weights (i.e. data sizes)
        weight_values = [t.weight for t in local_tensors]
        
        # so we can just use numpy.average
        return np.average(tensor_values, weights=weight_values, axis=0)

    
    def fets2022_1_aggregation(local_tensors,
                                 tensor_db,
                                 tensor_name,
                                 fl_round,
                                 collaborators_chosen_each_round,
                                 collaborator_times_per_round):
        if fl_round < _milestone:
            weight = [t.weight for t in local_tensors]
            tensor_values = [t.tensor for t in local_tensors]
            return np.average(tensor_values, weights=weight, axis=0)
        else:
            # 검색 조건 설정
            pre_tags = [(el, 'metric', 'validate_agg') for el in collaborators_chosen_each_round[fl_round]]
            post_tags = [(el, 'metric', 'validate_local') for el in collaborators_chosen_each_round[fl_round]]
            # search_tags = [('1', 'metric'), ('2', 'metric'), ('3', 'metric')]
    
            col_names = [t.col_name for t in local_tensors]
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
    
            past5_rounds = range(max(0, fl_round - 4), fl_round + 1)  # fl_round에서 4 라운드 이전까지
            # ***** el 이 이전 값에서 발견되어야함
            integral_loss_df = tensor_db[
                (tensor_db['tensor_name'] == 'valid_loss') &
                (tensor_db['round'].isin(past5_rounds)) &
                (tensor_db['tags'].apply(lambda x: x in post_tags)) &
                (tensor_db['origin'] == 'aggregator')
            ]
    
            integral_loss_dict = {k:0 for k in col_names}
            for idx, row in integral_loss_df.iterrows():
                k = row['tags'][0]
                integral_loss_dict[k] += float(row['nparray'])
            integ = [integral_loss_dict[col_name] for col_name in col_names]
            total_integ = sum(integ)
            integ = [el / total_integ for el in integ]
    
            weight = [t.weight for t in local_tensors]
    
            pre_cost = [pre_loss_dict[col_name] for col_name in col_names]
            post_cost = [post_loss_dict[col_name] for col_name in col_names]
            deriv = [max(0, pre - post) for (pre, post) in zip(pre_cost, post_cost)]
            total_deriv = sum(deriv) + 1e-10
            deriv = [el/total_deriv for el in deriv]
            PID = [0.45*w+0.1*m+0.45*k for (w, m, k) in zip(weight, integ, deriv)]
    
            tensor_values = [t.tensor for t in local_tensors]
            return np.average(tensor_values, weights=PID, axis=0)

    def fets2024_1_aggregation(local_tensors,
                                 tensor_db,
                                 tensor_name,
                                 fl_round,
                                 collaborators_chosen_each_round,
                                 collaborator_times_per_round):
        if fl_round < _milestone:
            weight = [t.weight for t in local_tensors]
            tensor_values = [t.tensor for t in local_tensors]
            return np.average(tensor_values, weights=weight, axis=0)
        else:
            # 검색 조건 설정
            pre_tags = [(el, 'metric', 'validate_agg') for el in collaborators_chosen_each_round[fl_round]]
            post_tags = [(el, 'metric', 'validate_local') for el in collaborators_chosen_each_round[fl_round]]
            # search_tags = [('1', 'metric'), ('2', 'metric'), ('3', 'metric')]
    
            col_names = [t.col_name for t in local_tensors]
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
    
            past5_rounds = range(max(0, fl_round - 4), fl_round + 1)  # fl_round에서 4 라운드 이전까지
            # ***** el 이 이전 값에서 발견되어야함
            integral_loss_df = tensor_db[
                (tensor_db['tensor_name'] == 'valid_loss') &
                (tensor_db['round'].isin(past5_rounds)) &
                (tensor_db['tags'].apply(lambda x: x in post_tags)) &
                (tensor_db['origin'] == 'aggregator')
            ]
    
            integral_loss_dict = {k:0 for k in col_names}
            for idx, row in integral_loss_df.iterrows():
                k = row['tags'][0]
                integral_loss_dict[k] += float(row['nparray'])
            integ = [integral_loss_dict[col_name] for col_name in col_names]
            total_integ = sum(integ)
            integ = [el / total_integ for el in integ]
    
            weight = [t.weight for t in local_tensors]
    
            pre_cost = [pre_loss_dict[col_name] for col_name in col_names]
            post_cost = [post_loss_dict[col_name] for col_name in col_names]
            deriv = [max(0, pre - post) for (pre, post) in zip(pre_cost, post_cost)]
            total_deriv = sum(deriv) + 1e-10
            deriv = [el/total_deriv for el in deriv]
            PID = [0.45*w+0.1*m+0.45*k for (w, m, k) in zip(weight, integ, deriv)]
    
            tensor_values = [t.tensor for t in local_tensors]
            return np.average(tensor_values, weights=deriv, axis=0)
    
    def fets2024_2_aggregation(local_tensors,
                                 tensor_db,
                                 tensor_name,
                                 fl_round,
                                 collaborators_chosen_each_round,
                                 collaborator_times_per_round):
        if fl_round < _milestone:
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

    
            # past5_rounds = range(max(0, fl_round - 4), fl_round + 1)  # fl_round에서 4 라운드 이전까지
            # integral_loss_df = tensor_db[
            #     (tensor_db['tensor_name'] == 'valid_loss') &
            #     (tensor_db['round'].isin(past5_rounds)) &
            #     (tensor_db['tags'].apply(lambda x: x in post_tags)) &
            #     (tensor_db['origin'] == 'aggregator')
            # ]
            # integral_loss_dict = {k:0 for k in col_names}
            # for idx, row in integral_loss_df.iterrows():
            #     k = row['tags'][0]
            #     integral_loss_dict[k] += float(row['nparray'])
            integ = [min(pre, post) + (max(0, pre - post)/2) for (pre, post) in zip(pre_cost, post_cost)]
            total_integ = sum(integ)
            integ = [el / total_integ for el in integ]

            
            weight = [t.weight for t in local_tensors]

            
            VPID = [0.45*w+0.1*m+0.45*k for (w, m, k) in zip(weight, integ, deriv)]
    
            tensor_values = [t.tensor for t in local_tensors]
            return np.average(tensor_values, weights=VPID, axis=0)
            
    # Adapted from FeTS Challenge 2021
    # Federated Brain Tumor Segmentation:Multi-Institutional Privacy-Preserving Collaborative Learning
    # Ece Isik-Polat, Gorkem Polat,Altan Kocyigit1, and Alptekin Temizel1
    def FedAvgM_Selection(local_tensors,
                          tensor_db,
                          tensor_name,
                          fl_round,
                          collaborators_chosen_each_round,
                          collaborator_times_per_round):
        
            """Aggregate tensors.
    
            Args:
                local_tensors(list[openfl.utilities.LocalTensor]): List of local tensors to aggregate.
                tensor_db: Aggregator's TensorDB [writable]. Columns:
                    - 'tensor_name': name of the tensor.
                        Examples for `torch.nn.Module`s: 'conv1.weight', 'fc2.bias'.
                    - 'round': 0-based number of round corresponding to this tensor.
                    - 'tags': tuple of tensor tags. Tags that can appear:
                        - 'model' indicates that the tensor is a model parameter.
                        - 'trained' indicates that tensor is a part of a training result.
                            These tensors are passed to the aggregator node after local learning.
                        - 'aggregated' indicates that tensor is a result of aggregation.
                            These tensors are sent to collaborators for the next round.
                        - 'delta' indicates that value is a difference between rounds
                            for a specific tensor.
                        also one of the tags is a collaborator name
                        if it corresponds to a result of a local task.
    
                    - 'nparray': value of the tensor.
                tensor_name: name of the tensor
                fl_round: round number
            Returns:
                np.ndarray: aggregated tensor
            """
            #momentum
            tensor_db.store(tensor_name='momentum',nparray=0.9,overwrite=False)
            #aggregator_lr
            tensor_db.store(tensor_name='aggregator_lr',nparray=1.0,overwrite=False)
    
            if fl_round == 0:
                # Just apply FedAvg
    
                tensor_values = [t.tensor for t in local_tensors]
                weight_values = [t.weight for t in local_tensors]               
                new_tensor_weight =  np.average(tensor_values, weights=weight_values, axis=0)        
    
                #if not (tensor_name in weight_speeds):
                if tensor_name not in tensor_db.search(tags=('weight_speeds',))['tensor_name']:    
                    #weight_speeds[tensor_name] = np.zeros_like(local_tensors[0].tensor) # weight_speeds[tensor_name] = np.zeros(local_tensors[0].tensor.shape)
                    tensor_db.store(
                        tensor_name=tensor_name, 
                        tags=('weight_speeds',), 
                        nparray=np.zeros_like(local_tensors[0].tensor),
                    )
                return new_tensor_weight        
            else:
                if tensor_name.endswith("weight") or tensor_name.endswith("bias"):
                    # Calculate aggregator's last value
                    previous_tensor_value = None
                    for _, record in tensor_db.iterrows():
                        if (record['round'] == fl_round 
                            and record["tensor_name"] == tensor_name
                            and record["tags"] == ("aggregated",)): 
                            previous_tensor_value = record['nparray']
                            break
    
                    if previous_tensor_value is None:
                        logger.warning("Error in fedAvgM: previous_tensor_value is None")
                        logger.warning("Tensor: " + tensor_name)
    
                        # Just apply FedAvg       
                        tensor_values = [t.tensor for t in local_tensors]
                        weight_values = [t.weight for t in local_tensors]               
                        new_tensor_weight =  np.average(tensor_values, weights=weight_values, axis=0)        
                        
                        if tensor_name not in tensor_db.search(tags=('weight_speeds',))['tensor_name']:    
                            tensor_db.store(
                                tensor_name=tensor_name, 
                                tags=('weight_speeds',), 
                                nparray=np.zeros_like(local_tensors[0].tensor),
                            )
    
                        return new_tensor_weight
                    else:
                        # compute the average delta for that layer
                        deltas = [previous_tensor_value - t.tensor for t in local_tensors]
                        weight_values = [t.weight for t in local_tensors]
                        average_deltas = np.average(deltas, weights=weight_values, axis=0) 
    
                        # V_(t+1) = momentum*V_t + Average_Delta_t
                        tensor_weight_speed = tensor_db.retrieve(
                            tensor_name=tensor_name,
                            tags=('weight_speeds',)
                        )
                        
                        momentum = float(tensor_db.retrieve(tensor_name='momentum'))
                        aggregator_lr = float(tensor_db.retrieve(tensor_name='aggregator_lr'))
                        
                        new_tensor_weight_speed = momentum * tensor_weight_speed + average_deltas # fix delete (1-momentum)
                        
                        tensor_db.store(
                            tensor_name=tensor_name, 
                            tags=('weight_speeds',), 
                            nparray=new_tensor_weight_speed
                        )
                        # W_(t+1) = W_t-lr*V_(t+1)
                        new_tensor_weight = previous_tensor_value - aggregator_lr*new_tensor_weight_speed
    
                        return new_tensor_weight
                else:
                    # Just apply FedAvg       
                    tensor_values = [t.tensor for t in local_tensors]
                    weight_values = [t.weight for t in local_tensors]               
                    new_tensor_weight =  np.average(tensor_values, weights=weight_values, axis=0)
    
                    return new_tensor_weight
    
    
    # # Running the Experiment
    # 
    # ```run_challenge_experiment``` is singular interface where your custom methods can be passed.
    # 
    # - ```aggregation_function```, ```choose_training_collaborators```, and ```training_hyper_parameters_for_round``` correspond to the [this list](#Custom-hyperparameters-for-training) of configurable functions 
    # described within this notebook.
    # - ```institution_split_csv_filename``` : Describes how the data should be split between all collaborators. Extended documentation about configuring the splits in the ```institution_split_csv_filename``` parameter can be found in the [README.md](https://github.com/FETS-AI/Challenge/blob/main/Task_1/README.md). 
    # - ```db_store_rounds``` : This parameter determines how long metrics and weights should be stored by the aggregator before being deleted. Providing a value of `-1` will result in all historical data being retained, but memory usage will likely increase.
    # - ```rounds_to_train``` : Defines how many rounds will occur in the experiment
    # - ```device``` : Which device to use for training and validation
    
    # ## Setting up the experiment
    # Now that we've defined our custom functions, the last thing to do is to configure the experiment. The following cell shows the various settings you can change in your experiment.
    # 
    # Note that ```rounds_to_train``` can be set as high as you want. However, the experiment will exit once the simulated time value exceeds 1 week of simulated time, or if the specified number of rounds has completed.
    
    
    # change any of these you wish to your custom functions. You may leave defaults if you wish.
    aggregation_function = fets2022_1_aggregation # weighted_average_aggregation# FedAvgM_Selection  # weighted_average_aggregation
    choose_training_collaborators = subset_selection_for_collaborators_on_rounds # all_collaborators_train
    training_hyper_parameters_for_round = major_minor_parameters # constant_hyper_parameters
    
    # As mentioned in the 'Custom Aggregation Functions' section (above), six 
    # perfomance evaluation metrics are included by default for validation outputs in addition 
    # to those you specify immediately above. Changing the below value to False will change 
    # this fact, excluding the three hausdorff measurements. As hausdorff distance is 
    # expensive to compute, excluding them will speed up your experiments.
    include_validation_with_hausdorff=False
    
    # We encourage participants to experiment with partitioning_1 and partitioning_2, as well as to create
    # other partitionings to test your changes for generalization to multiple partitionings.
    #institution_split_csv_filename = 'partitioning_1.csv'
    # institution_split_csv_filename = 'small_split.csv'
    # institution_split_csv_filename = 'FeTS2_stage1_2.csv'
    
    # change this to point to the parent directory of the data
    # brats_training_data_parent_dir = '/home2/dwnusa/2024_data/FeTS2022/center'
    
    # increase this if you need a longer history for your algorithms
    # decrease this if you need to reduce system RAM consumption
    
    # this is passed to PyTorch, so set it accordingly for your system
    # device = 'cuda'
    
    # you'll want to increase this most likely. You can set it as high as you like, 
    # however, the experiment will exit once the simulated time exceeds one week. 
    # rounds_to_train = 20
    
    # (bool) Determines whether checkpoints should be saved during the experiment. 
    # The checkpoints can grow quite large (5-10GB) so only the latest will be saved when this parameter is enabled
    
    # path to previous checkpoint folder for experiment that was stopped before completion. 
    # Checkpoints are stored in ~/.local/workspace/checkpoint, and you should provide the experiment directory 
    # relative to this path (i.e. 'experiment_1'). Please note that if you restore from a checkpoint, 
    # and save checkpoint is set to True, then the checkpoint you restore from will be subsequently overwritten.
    # restore_from_checkpoint_folder = 'experiment_1'
    
    
    # the scores are returned in a Pandas dataframe
    scores_dataframe, checkpoint_folder = run_challenge_experiment(
        aggregation_function=aggregation_function,
        choose_training_collaborators=choose_training_collaborators,
        training_hyper_parameters_for_round=training_hyper_parameters_for_round,
        include_validation_with_hausdorff=include_validation_with_hausdorff,
        institution_split_csv_filename=institution_split_csv_filename,
        brats_training_data_parent_dir=brats_training_data_parent_dir,
        db_store_rounds=db_store_rounds,
        rounds_to_train=rounds_to_train,
        device=device,
        save_checkpoints=save_checkpoints,
        restore_from_checkpoint_folder = restore_from_checkpoint_folder)
    
    
    scores_dataframe

elif sys.argv[1] == 'infer':
    print(type(sys.argv[2]), sys.argv[2]=='str')
    assert isinstance(sys.argv[2], str), f"argv[2] required for checkpoint_folder"
    # ## Produce NIfTI files for best model outputs on the validation set
    # Now we will produce model outputs to submit to the leader board.
    # 
    # At the end of every experiment, the best model (according to average ET, TC, WT DICE) 
    # is saved to disk at: ~/.local/workspace/checkpoint/\<checkpoint folder\>/best_model.pkl,
    # where \<checkpoint folder\> is the one printed to stdout during the start of the 
    # experiment (look for the log entry: "Created experiment folder experiment_##..." above).
    
    
    from fets_challenge import model_outputs_to_disc
    from pathlib import Path
    
    # infer participant home folder
    home = str(Path.home())
    
    # you will need to specify the correct experiment folder and the parent directory for
    # the data you want to run inference over (assumed to be the experiment that just completed)
    
    checkpoint_folder=f'{sys.argv[2]}'# 'experiment_'
    os.makedirs(os.path.join(home, f'.local/{workspace}/checkpoint', checkpoint_folder), exist_ok=True)
    assert os.path.isdir(os.path.join(home, f'.local/{workspace}/checkpoint', checkpoint_folder)), f"{sys.argv[2]} not exist"
    #data_path = </PATH/TO/CHALLENGE_VALIDATION_DATA>
    data_path = brats_training_data_parent_dir
    # validation_csv_filename = 'validation.csv'
    
    # you can keep these the same if you wish
    final_model_path = os.path.join(home, f'.local/{workspace}/checkpoint', checkpoint_folder, 'best_model.pkl')
    
    # If the experiment is only run for a single round, use the temp model instead
    if not Path(final_model_path).exists():
       final_model_path = os.path.join(home, f'.local/{workspace}/checkpoint', checkpoint_folder, 'temp_model.pkl')
    
    outputs_path = os.path.join(home, f'.local/{workspace}/checkpoint', checkpoint_folder, 'model_outputs')
    
    
    # Using this best model, we can now produce NIfTI files for model outputs 
    # using a provided data directory
    
    model_outputs_to_disc(data_path=data_path, 
                          validation_csv=validation_csv_filename,
                          output_path=outputs_path, 
                          native_model_path=final_model_path,
                          outputtag='',
                          device=device)
else:
    print(f"{sys.argv[1]} is unknown on {os.getlogin()}")
