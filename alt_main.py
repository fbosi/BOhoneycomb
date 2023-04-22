import os
import sys
import time
import uuid
import yaml
from pathlib import Path
import pprint

import concurrent.futures

from ax import *
from ax.metrics.noisy_function import NoisyFunctionMetric
from ax.service.utils.report_utils import exp_to_df
from ax.runners.synthetic import SyntheticRunner

from ax.modelbridge.factory import get_MOO_EHVI

from ax.modelbridge.modelbridge_utils import observed_hypervolume


from utils.project_utils import Simulation, set_up_dirs, clean_replay, get_datestring
import utils.plot_utils

import numpy as np
import pandas as pd

# Define the parameters
eta = RangeParameter(name = "eta", parameter_type = ParameterType.FLOAT, lower=0.1, upper=0.95)
xi = RangeParameter(name = "xi", parameter_type = ParameterType.FLOAT, lower=0.1, upper=0.95)


# Define the search space
search_space = SearchSpace(
    parameters = [eta, xi],
    )



model_name = 'hex_solid_thick_buckle'
script_name = model_name

result_metrics = ['stress_ratio', 'stiffness_ratio']

# Directories
temp_dir  = set_up_dirs('run','_temp')
runfile_dir = set_up_dirs('run','runfiles')

sim = Simulation(model_name,script_name,result_metrics, temp_dir, runfile_dir)
    
class MetricA(NoisyFunctionMetric):
    def f(self,x: dict) -> float:
        return float(sim.get_results(x)[self.name])

class MetricB(NoisyFunctionMetric):
    def f(self, x: dict) -> float:
        return float(sim.get_results(x)[self.name])

metric_a = MetricA("stress_ratio",["eta","xi"],noise_sd = 0.0, lower_is_better=True)
metric_b = MetricB("stiffness_ratio",["eta","xi"],noise_sd = 0.0, lower_is_better=True)

mo = MultiObjective(
    objectives = [Objective(metric=metric_a), Objective(metric=metric_b)],
    )

objective_thresholds = [ObjectiveThreshold(metric, bound=0.9, relative=False) for metric in mo.metrics]

optimization_config = MultiObjectiveOptimizationConfig(
    objective=mo,
    objective_thresholds=objective_thresholds,
    )

N_INIT = 6
N_BATCH = 25

def build_experiment():
    experiment = Experiment(
        name = "pareto_experiment",
        search_space = search_space,
        optimization_config=optimization_config,
        runner=SyntheticRunner(),
        )
    return experiment

def initialize_experiment(experiment):
    sobol = Models.SOBOL(search_space=experiment.search_space, seed=1234)
    
    for _ in range(N_INIT):
        experiment.new_trial(sobol.gen(1)).run()
        
    return experiment.fetch_data()

sobol_experiment = build_experiment()
sobol_data = initialize_experiment(sobol_experiment)

sobol_model = Models.SOBOL(experiment=sobol_experiment, data=sobol_data)

sobol_hv_list = []

for i in range(N_BATCH):
    generator_run = sobol_model.gen(1)
    trial = sobol_experiment.new_trial(generator_run=generator_run)
    trial.run()
    exp_df = exp_to_df(sobol_experiment)
    outcomes = np.array(exp_df[['xi','eta']], dtype=np.double)
    
    # Fit a GP based model in order to calculate hypervolume (not to generate new points)
    
    dummy_model = get_MOO_EHVI(
        experiment = sobol_experiment,
        data = sobol_experiment.fetch_data(),
        )
    try:
        hv = observed_hypervolume(modelbridge=dummy_model)
    except:
        hv=0
        print("Failed to compute hv")
    sobol_hv_list.append(hv)
    print(f"Iteration: {i}, HV: {hv}")

sobol_outcomes = np.array(exp_to_df(sobol_experiment)[['xi','eta']],dtype=np.double)