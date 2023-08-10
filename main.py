"""
Main simulation script for running the optimization process.

Overview: This script sets up and executes a simulation based on the specified
model name and associated configuration. Most of the configuration is handled
via YAML files, and specific Abaqus Python scripts are used to obtain the
results.

Instructions:
- Model Selection: Modify the 'model_name' variable to select the
                   desired model. If the abaqus installation is not available
                   from the command line when calling 'abaqus', also modify the
                   'abaqus_dir' variable to point to the abaqus installation.
- Configuration: All other settings, parameters, and
                 configurations should be handled through the corresponding YAML
                 files. No further modifications to this script are typically
                 required.

Note: It is highly recommended not to modify this file beyond the model
selection unless you are aware of the specific changes needed for your use case.
"""
import os

import time
import yaml
import concurrent.futures
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
from ax.modelbridge.generation_strategy import (GenerationStep,
                                                GenerationStrategy)
from ax.modelbridge.registry import Models

from utils.project_utils import (Simulation, set_up_dirs, get_datestring)
import utils.plot_utils

#                                                               MODEL SELECTION
# ==============================================================================
# Select the model in models/ directory to run the simulation
model_name = 'hex_solid_tanh_new'
# Create the config file name based on the model name
config_file = f'{model_name}_config.yml'

# Directory used to run abaqus in command line (Windows)
abaqus_dir = 'abaqus'

#                                                                 CONFIGURATION
# ==============================================================================
try:
    config_dir = os.path.join(os.getcwd(), 'run', 'models', model_name,
                              config_file)
except Exception:
    raise FileNotFoundError(
        'Cannot find the configuration file for the selected model')

# Read the optimization configuration from a YAML file
with open(config_dir) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# Extract the relevant parameters and settings
script_name = config['script_name']  # script name
opt_config = config['optimisation']  # optimisation setup

# Set up/ validate the folder structure used by the program
temp_dir = set_up_dirs('run', '_temp')
runfile_dir = set_up_dirs('run', 'runfiles')
db_dir = set_up_dirs('database')

# Multiobjective flag
multiobjective = opt_config['multiobjective']

#                                                                          MAIN
# =============================================================================


def main():
    if multiobjective:
        objective_config = opt_config['multi']
        # result_metrics is a list of objective and constraint names
        result_metrics = opt_config['constraint_metrics'] + [
            i.get('name') for i in objective_config['objective_metrics']
        ]
    else:
        objective_config = opt_config['single']
        result_metrics = opt_config['constraint_metrics'] + [
            objective_config['objective_metric']
        ]

    # Instantiate the Simulation object which handles the simulation process
    # in ABAQUS
    sim = Simulation(model_name, script_name, result_metrics, temp_dir,
                     runfile_dir, abaqus_dir)

    # Clean up runfile and _temp before running a simulation
    sim.clean_up_prompt()

    #                                                            BO IN SERVICE API
    # =============================================================================
    # Set up the generation strategy
    NUM_SOBOL_STEPS = opt_config['num_sobol_steps']
    NUM_OF_ITERS = opt_config['num_of_iters']
    BATCH_SIZE = 1  # Run BO sequentially

    gs = GenerationStrategy(steps=[
        GenerationStep(model=Models.SOBOL, num_trials=NUM_SOBOL_STEPS),
        GenerationStep(
            model=Models[opt_config['model']],
            num_trials=-1,
        )
    ])

    # Initialize the AxClient
    ax_client = AxClient(generation_strategy=gs,
                         random_seed=123,
                         verbose_logging=True)

    # Define parameters
    if opt_config['uniform_params']:  # for uniform parameter range
        params = [{
            'name': f'x{i + 1}',
            'type': 'range',
            'bounds': [opt_config['lo_bound'], opt_config['up_bound']],
            'value_type': 'float'
        } for i in range(opt_config['num_of_params'])]

    else:
        params = opt_config['parameters']

    # Create an experiment
    if multiobjective:
        ax_client.create_experiment(
            name=opt_config['experiment_name'],
            parameters=params,
            objectives={
                i['name']:
                ObjectiveProperties(minimize=i['minimize'],
                                    threshold=i['threshold'])
                for i in objective_config['objective_metrics']
            },
            outcome_constraints=opt_config['outcome_constraints'],
            parameter_constraints=opt_config['parameter_constraints'])
    else:
        ax_client.create_experiment(
            name=opt_config['experiment_name'],
            parameters=params,
            objective_name=objective_config['objective_metric'],
            minimize=objective_config[
                'minimize'],  # Optional, defaults to False.
            outcome_constraints=opt_config['outcome_constraints'],
            parameter_constraints=opt_config['parameter_constraints'])

    abandoned_trials_count = 0
    NUM_OF_BATCHES = NUM_OF_ITERS // BATCH_SIZE if NUM_OF_ITERS % BATCH_SIZE == 0 else NUM_OF_ITERS // BATCH_SIZE + 1

    for i in range(NUM_OF_BATCHES):
        try:
            results = {}
            trials_to_evaluate = {}
            # Sequentially generate the batch
            for j in range(min(NUM_OF_ITERS - i * BATCH_SIZE, BATCH_SIZE)):
                parameterization, trial_index = ax_client.get_next_trial()
                trials_to_evaluate[trial_index] = parameterization

            # Evaluate the results in parallel and append results to a dictionary
            for trial_index, parametrization in trials_to_evaluate.items():
                with concurrent.futures.ProcessPoolExecutor(
                        max_workers=3) as executor:
                    try:
                        # The evaluation function is set to the anlytical
                        # function defined in the Simulation class - for use
                        # with ABAQUS, change this to the get_results method
                        eval_func = sim.get_results
                        exec = executor.submit(eval_func, parametrization)
                        results.update({trial_index: exec.result()})
                    except Exception as e:
                        ax_client.abandon_trial(trial_index=trial_index)
                        abandoned_trials_count += 1
                        print(
                            f'[WARNING] Abandoning trial {trial_index} due to processing errors.'
                        )
                        print(e)
                        if abandoned_trials_count > 0.1 * NUM_OF_ITERS:
                            print(
                                '[WARNING] More than 10 % of iterations were abandoned. Consider improving the '
                                'parametrization.')

            for trial_index in results:
                ax_client.complete_trial(trial_index, results.get(trial_index))

        except KeyboardInterrupt:
            print('Program interrupted by user')
            break

    try:
        # Save `AxClient` to a JSON snapshot.
        _, dt_string = get_datestring()

        db_save_name = f'simulation_run_{dt_string}.json'
        ax_client.save_to_json_file(
            filepath=os.path.join(db_dir, db_save_name))

    except Exception:
        print(
            '[WARNING] The JSON snapshot of the Ax Client has not been saved.')

    return ax_client


if __name__ == "__main__":

    load_existing_client = False
    client_filename = 'FINAL_RD10.json'
    client_filepath = os.path.join(db_dir, client_filename)

    start = time.perf_counter()
    if load_existing_client:
        # (Optional) Reinstantiate an `AxClient` from a JSON snapshot.
        ax_client = AxClient.load_from_json_file(filepath=client_filepath)
    else:
        # Run the simulation
        ax_client = main()

    finish = time.perf_counter()

    print(f'Simulation took {finish - start} seconds to complete')

    # Plotting

    save_pdf = True
    save_png = True  # This must be true for generating reports
    plot_dir = os.path.join(os.getcwd(), 'reports', 'plots')

    P = utils.plot_utils.Plot(ax_client, plot_dir, save_pdf, save_png)

    if multiobjective:
        try:
            P.plot_moo_trials()
            #params = P.plot_posterior_pareto_frontier()

        except Exception as e:
            print('[WARNING] An exception occured while plotting!')
            print(e)
    else:
        try:
            P.plot_single_objective_trials()
            P.plot_single_objective_convergence()
            P.plot_single_objective_distances()
        except Exception as e:
            print('[WARNING] An exception occured while plotting!')
            print(e)
