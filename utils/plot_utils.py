from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
#mpl.use('Agg')
import numpy as np
from scipy.stats import norm
import os

from ax.plot.pareto_utils import compute_posterior_pareto_frontier, get_observed_pareto_frontiers
from ax.plot.pareto_frontier import plot_pareto_frontier
from ax.plot.contour import _get_contour_predictions
from ax.plot.helper import _format_dict


# from .project_utils import clean_directory


class Plot:

    def __init__(self, ax_client, plot_dir, save_pdf=False, save_png=False):

        self.ax_client = ax_client
        self.experiment = ax_client.experiment
        self.objective = ax_client.experiment.optimization_config.objective
        self.trials_df = ax_client.get_trials_data_frame().sort_values(by=['trial_index'], ascending=True)
        self.trial_values = self.experiment.trials.values()
        self.sobol_num = ax_client.generation_strategy._steps[0].num_trials

        self.plot_dir = plot_dir
        Path(plot_dir).mkdir(parents=True, exist_ok=True)  # check if plot directory exists

        self.save_pdf = save_pdf
        self.save_png = save_png

        # Plot style setup
        mpl.rcParams.update(mpl.rcParamsDefault)  # reset style
        if not mpl.is_interactive(): plt.ion()  # enable interactive mode
        # plt.style.use('dark_background')
        plt.rcParams['font.size'] = 18
        plt.rcParams['axes.labelweight'] = 'bold'
        plt.rcParams['axes.linewidth'] = 2
        plt.rcParams['axes.axisbelow'] = True
        plt.rcParams["figure.figsize"] = (10, 6)
        plt.rcParams['text.usetex'] = False

    # Single Objective Plotting

    def plot_single_objective_convergence(self):
        # Calculations
        best_objectives = np.array([[trial.objective_mean for trial in self.experiment.trials.values()]])

        if self.objective.minimize:
            y = np.minimum.accumulate(best_objectives, axis=1)
        else:
            y = np.maximum.accumulate(best_objectives, axis=1)

        x = np.arange(1, np.size(y) + 1)

        # Plotting
        fig = plt.figure()

        plt.plot(x, y.T, linewidth=3)
        plt.axvline(self.sobol_num, linewidth=2, linestyle='--', color='k')

        plt.title('Convergence plot')
        plt.xlabel('Trial #')
        plt.ylabel('Best objective')
        plt.ylim(*plt.ylim())
        plt.xlim([1, np.size(y) + 1])

        self.save_plot('convergence_plot', fig)
        plt.show()

    def plot_single_objective_trials(self):
        # Consecutive evaluations

        objective_name = self.objective.metric.name

        values = np.asarray([trial.get_metric_mean(objective_name)
                             if trial.status.is_completed
                             else np.nan
                             for trial in self.trial_values])

        x = np.arange(1, np.size(values) + 1)
        # Plotting
        fig = plt.figure()

        plt.plot(x, values.T, marker='.', markersize=20, linewidth=3)
        plt.axvline(self.sobol_num, linewidth=2, linestyle='--', color='k')
        plt.title('Consecutive evaluations plot')
        plt.xlabel('Trial #')
        plt.ylabel('Objective value')
        plt.ylim(*plt.ylim())
        plt.xlim([1, np.size(values) + 1])

        self.save_plot('evaluations_plot', fig)
        plt.show()

    def plot_single_objective_distances(self):
        arms_by_trial = np.array([list(trial.arm.parameters.values())
                                  for trial in self.trial_values])

        # Distances between evaluations
        distances = np.linalg.norm(np.diff(arms_by_trial, axis=0), ord=2, axis=1)

        fig = plt.figure()

        plt.plot(np.arange(0, len(distances)), distances, linewidth=3, marker='.', markersize=20)
        plt.axvline(self.sobol_num, linewidth=2, linestyle='--', color='k')

        plt.title('Distances plot')
        plt.xlabel('Trial #')
        plt.ylabel('Distance |x[n]-x[n-1]|')

        self.save_plot('distances_plot', fig)
        plt.show()

    def plot_contour_plt(self, param_x, param_y, metric_name, density=50):

        best_parameters, values = self.ax_client.get_best_parameters()
        model = self.ax_client.generation_strategy.model

        data, f_plt, sd_plt, grid_x, grid_y, scales = _get_contour_predictions(
            model=model,
            x_param_name=param_x,
            y_param_name=param_y,
            metric=metric_name,
            generator_runs_dict=None,
            density=density)

        X, Y = np.meshgrid(grid_x, grid_y)
        Z_f = np.asarray(f_plt).reshape(density, density)
        Z_sd = np.asarray(sd_plt).reshape(density, density)

        labels = []
        evaluations = []

        for key, value in data[1].items():
            labels.append(key)
            evaluations.append(list(value[1].values()))

        evaluations = np.asarray(evaluations)
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=self.figsize)
        cont1 = axes[0].contourf(X, Y, Z_f, 20, cmap='viridis')
        fig.colorbar(cont1, ax=axes[0])
        axes[0].set_title('Mean')
        axes[0].plot(evaluations[:, 0],
                     evaluations[:, 1],
                     'o',
                     markersize=12,
                     mfc='white',
                     mec='black')

        axes[0].plot(best_parameters[param_x],
                     best_parameters[param_y],
                     'o',
                     markersize=13,
                     mfc='red',
                     mec='black')

        cont2 = axes[1].contourf(X, Y, Z_sd, 20, cmap='plasma')
        fig.colorbar(cont2, ax=axes[1])
        axes[1].set_title('Standard Deviation')
        axes[1].plot(evaluations[:, 0],
                     evaluations[:, 1],
                     'o',
                     markersize=12,
                     mfc='white',
                     mec='black')

        axes[1].plot(best_parameters[param_x],
                     best_parameters[param_y],
                     'o',
                     markersize=13,
                     mfc='red',
                     mec='black')

        for axs in axes.flat:
            axs.set(xlabel=param_x, ylabel=param_y)

        fig.tight_layout()

        self.save_plot(plt, 'contours_plot')

    ## Multiple Objective Plotting
    def plot_moo_trials(self, objective_labels=None):
        objective_names = [i.metric.name for i in self.objective.objectives]

        fig, axes = plt.subplots()
        df = self.trials_df
        objective_values = {i: df.get(i).values for i in objective_names}
        x, y = objective_values.values()

        axes.scatter(x, y, s=70, c=df.index, cmap='viridis')  # All trials
        fig.colorbar(axes.collections[0], ax=axes, label='trial #')

        # for idx, label in enumerate(df.index.values):
        #     axes.annotate(label, (x[idx], y[idx]))
        if objective_labels is None:
            plt.xlabel(objective_names[0])
            plt.ylabel(objective_names[1])
            
        else:
            plt.xlabel(objective_labels[0])
            plt.ylabel(objective_labels[1])
        
        axes.set_title('Consecutive MOO Trials')
        fig.tight_layout()
        self.save_plot('consecutive_moo_plot', fig)
        #plt.show()

    def plot_posterior_pareto_frontier(self):

        objective_names = [i.metric.name for i in self.objective.objectives]
        frontier = compute_posterior_pareto_frontier(
            experiment=self.experiment,
            data=self.experiment.fetch_data(),
            primary_objective=self.objective.objectives[0].metric,
            secondary_objective=self.objective.objectives[1].metric,
            absolute_metrics=objective_names,  # we choose all metrics
            num_points=10,  # number of points in the pareto frontier
        )
        all_metrics = frontier.means.keys()
        # Parametrization list to retrieve
        labels = []
        if frontier.arm_names is None:
            arm_names = [f"Parameterization {i}" for i in range(len(frontier.param_dicts))]
        else:
            arm_names = [f"Arm {name}" for name in frontier.arm_names]

        for i, param_dict in enumerate(frontier.param_dicts):
            label = []
            for metric in all_metrics:
                label.append(f'metric: {metric}, {frontier.means[metric][i]}, ')
            label.append(f'parametrization:{param_dict}')
            labels.append(label)

        fig, axes = plt.subplots()
        axes.scatter(*[frontier.means[i] for i in objective_names], s=70, c='k')  # Pareto front
        

        plt.xlabel(objective_names[0])
        plt.ylabel(objective_names[1])
        axes.set_title('Posterior Pareto Frontier')

        fig.tight_layout()
        self.save_plot('pareto_plot', fig)
        plt.show()
        return labels
    # def plot_posterior_pareto_frontier(self):
        # objective_names = [i.metric.name for i in self.objective.objectives]
        # frontier = compute_posterior_pareto_frontier(
            # experiment=self.experiment,
            # data=self.experiment.fetch_data(),
            # primary_objective=self.objective.objectives[0].metric,
            # secondary_objective=self.objective.objectives[1].metric,
            # absolute_metrics=objective_names,
            # num_points=100,
        # )
        
        # # Extract mean and standard deviation for each metric
        # means = {metric: frontier.means[metric] for metric in objective_names}
        # stds = {metric: np.sqrt(frontier.covariance[metric, metric].reshape(-1, 1)) 
                # for metric in objective_names}
        
        # # Get the labels for the points
        # all_metrics = frontier.means.keys()
        # labels = []
        # if frontier.arm_names is None:
            # arm_names = [f"Parameterization {i}" for i in range(len(frontier.param_dicts))]
        # else:
            # arm_names = [f"Arm {name}" for name in frontier.arm_names]
        # for i, param_dict in enumerate(frontier.param_dicts):
            # label = []
            # for metric in all_metrics:
                # label.append(f'metric: {metric}, {frontier.means[metric][i]}, ')
            # label.append(f'parametrization:{param_dict}')
            # labels.append(label)

        # fig, axes = plt.subplots()
        # # Add error bars to the scatter plot
        # axes.errorbar(x=means[objective_names[0]], y=means[objective_names[1]], 
                      # xerr=stds[objective_names[0]], yerr=stds[objective_names[1]], 
                      # fmt='o', markersize=7, color='k', ecolor='gray', capsize=4)
        # plt.xlabel(objective_names[0])
        # plt.ylabel(objective_names[1])
        # axes.set_title('Posterior Pareto Frontier')
        # fig.tight_layout()
        # self.save_plot('pareto_plot', fig)
        # #plt.show()
        # return labels

    # # # Plot utilities
    # # def clean_plot_dir(self):
    # #     clean_directory(self.plot_dir)

    def save_plot(self, name, fig):
        save_name = os.path.join(self.plot_dir, name)
        if self.save_pdf:
            fig.savefig(f'{save_name}.eps', transparent=False, bbox_inches='tight')

        if self.save_png:
            fig.savefig(f'{save_name}.png', transparent=False, bbox_inches='tight')
