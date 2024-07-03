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
from fets_challenge import model_outputs_to_disc
from pathlib import Path

workspace = 'workspace'
brats_training_data_parent_dir = f'/home2/{os.getlogin()}/2024_data/FeTS2022/center'
assert os.path.isdir(brats_training_data_parent_dir), f"not exist folder {brats_training_data_parent_dir}"
device = 'cuda'
validation_csv_filename = 'validation.csv'

assert isinstance(sys.argv[2], str), f"argv[2] required for checkpoint_folder"

# infer participant home folder
home = str(Path.home())
checkpoint_folder=f'{sys.argv[2]}'
os.makedirs(os.path.join(home, f'.local/{workspace}/checkpoint', checkpoint_folder), exist_ok=True)
assert os.path.isdir(os.path.join(home, f'.local/{workspace}/checkpoint', checkpoint_folder)), f"{sys.argv[2]} not exist"
data_path = brats_training_data_parent_dir
final_model_path = os.path.join(home, f'.local/{workspace}/checkpoint', checkpoint_folder, 'best_model.pkl')

if not Path(final_model_path).exists():
    final_model_path = os.path.join(home, f'.local/{workspace}/checkpoint', checkpoint_folder, 'temp_model.pkl')

outputs_path = os.path.join(home, f'.local/{workspace}/checkpoint', checkpoint_folder, 'model_outputs')

model_outputs_to_disc(data_path=data_path, 
                        validation_csv=validation_csv_filename,
                        output_path=outputs_path, 
                        native_model_path=final_model_path,
                        outputtag='',
                        device=device)
