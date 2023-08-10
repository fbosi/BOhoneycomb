from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from fpdf import FPDF
import os

import uuid
import sys
from sys import exit
import glob
import subprocess
import pickle
import numpy as np
from random import randrange


# Datestrings used in filenames
def get_datestring():
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y at %H:%M:%S")
    dt_string2 = now.strftime('%d%m%Y%H%M%S')

    return dt_string, dt_string2


# Standalone functions
def yes_or_no(question):
    while "the answer is invalid":
        reply = str(input(question + ' (y/n): ')).lower().strip()
        if reply[:1] == 'y':
            return True
        if reply[:1] == 'n':
            return False


def clean_directory(dir_path):
    [f.unlink() for f in Path(dir_path).glob("*") if f.is_file()]


def clean_replay():
    directory = os.path.dirname(os.path.realpath(__file__))

    filelist = os.listdir(directory)

    for item in filelist:
        if '.rpy' in item or '.rec' in item:
            try:
                os.remove(os.path.join(directory, item))
            except OSError:
                pass


# Set up the directories used by the program
def set_up_dirs(*arg):
    """Helper function for setting up directories in current dir
    and validating the folder structure"""
    dir_path = os.path.abspath(os.path.join(os.getcwd(), *arg))
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    return dir_path


def generate_report(opt_config, means, best_parameters):
    pagewidth = 210
    margin = 50

    dt_string, dt_string2 = get_datestring()

    def create_title(pdf):
        pdf.set_font('Helvetica', 'B', 24)
        pdf.ln(15)
        pdf.write(5, 'Bayesian Optimization Report')
        pdf.ln(10)
        pdf.set_font('Helvetica', '', 16)
        pdf.write(5, f'Generated on {dt_string}')
        pdf.line(10, 45, 200, 45)
        pdf.ln(5)

    def create_section(pdf, section_title):
        pdf.ln(10)
        pdf.set_font('Helvetica', 'BU', 16)
        pdf.write(5, section_title)
        pdf.set_font('Helvetica', '', 12)
        pdf.ln(7)

    def create_subsection(pdf, section_title):
        pdf.set_font('Helvetica', 'B', 14)
        pdf.write(5, section_title)
        pdf.set_font('Helvetica', '', 12)
        pdf.ln(7)

    def create_textblock(pdf, line_dict):
        pdf.set_font('Helvetica', '', 12)
        for k, v in line_dict.items():
            pdf.set_font('Helvetica', 'BI', 12)
            pdf.write(5, f'{k}: ')
            pdf.set_font('Helvetica', '', 12)
            pdf.write(5, f'{v}')
            pdf.ln(5)
        pdf.ln(5)

    def draw_plots(pdf, plot_paths):

        plots = glob.glob(plot_paths)

        for idx, plot in enumerate(plots, start=1):
            plot_name = os.path.basename(plot).split('.')[0]
            create_subsection(pdf, plot_name)
            pdf.image(plot, x=margin / 2, y=None, w=pagewidth - margin)
            if idx % 2 == 0:
                pdf.add_page()
            else:
                pdf.ln(10)

    pdf = FPDF()

    plot_paths = os.path.join(os.getcwd(), 'reports', 'plots', '*.png')
    pdf.add_page()
    create_title(pdf)
    create_section(pdf, 'Configuration Data')
    create_textblock(pdf, opt_config)
    create_section(pdf, 'Optimisation Results')
    create_textblock(pdf, means)
    create_textblock(pdf, best_parameters)
    pdf.add_page()
    create_section(pdf, 'Plots')
    draw_plots(pdf, plot_paths)

    report_name = f'report_{dt_string2}.pdf'
    report_dir = os.path.join(os.getcwd(), 'reports')
    report_path = os.path.join(report_dir, report_name)

    Path(report_dir).mkdir(parents=True, exist_ok=True)
    pdf.output(report_path, 'F')


# Simulation class
class Simulation:
    """
    A class to represent and manage simulations for a given model in ABAQUS.

    Attributes
    ----------
    model_name : str
        Name of the model used for simulation.
    script_name : str
        Name of the script to run the simulation.
    result_metrics : object
        Metrics used for the result analysis.
    temp_dir : str
        Temporary directory path for storing temporary simulation files.
    runfile_dir : str
        Directory path to save run files.
    abaqus_dir : str, optional
        Directory containing ABAQUS installation (default is 'abaqus').
    save_cae : bool, optional
        Flag to determine whether to save the CAE file or not (default is
        False).

    Methods
    -------
    clean_up_prompt():
        Prompts the user for cleaning up previous simulations.
    get_results(parametrization: dict) -> dict:
        Runs the simulation with the given parametrization and retrieves the
        results.
    analytical_hex_thick(parametrization: dict) -> dict:
        Placeholder function wiht the same signature as get_results for testing
        the BO.

    Examples
    --------
    >>> simulation = Simulation("my_model", "script_name", metrics, "temp_dir", "runfile_dir")
    >>> parametrization = {'eta': 0.5, 'xi': 0.3}
    >>> simulation.get_results(parametrization)
    """

    def __init__(self,
                 model_name,
                 script_name,
                 result_metrics,
                 temp_dir,
                 runfile_dir,
                 abaqus_dir='abaqus',
                 save_cae=False):

        # Setting up directories
        self.model_name = model_name
        self.script_name = script_name
        self.temp_dir = temp_dir
        self.runfile_dir = runfile_dir
        self.result_metrics = result_metrics
        self.save_cae = save_cae
        self.abaqus_dir = abaqus_dir
        self.iterator = 0

        self.check_abaqus_dir()

    def check_abaqus_dir(self):

        msg = (f"The specified abaqus_dir: '{self.abaqus_dir}' is incorrect or "
               "no Abaqus installation was found on the system. If testing the "
               "BO, please use Simulation.analytical_hex_thick instead of "
               "Simulation.get_results as the evaluation function")
        # Check for the existence of Abaqus installation
        try:
            response = subprocess.run(self.abaqus_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print("Output:", response.stdout.decode())
        except FileNotFoundError:
            raise FileNotFoundError(msg)
        except Exception as e:
            raise Exception(msg)

    def clean_up_prompt(self):
        """
        Prompts the user to clean up all previous simulations. If the user accepts,
        it cleans the directories; otherwise, it terminates the script.
        """
        if yes_or_no(
                "Running the program will delete all previous"
                " simulation results. Proceed?"
        ):
            clean_directory(self.temp_dir)
            clean_directory(self.runfile_dir)
        else:
            print('Terminating the script')
            exit()

    def get_results(self, parametrization):
        """
        Runs the simulation script with the given parametrization and retrieves
        the results. This method can be used as the evaluation function of
        the AxClient to perform BO using Abaqus.

        Parameters
        ----------
        parametrization : dict
            Dictionary containing parameters for the simulation.

        Returns
        -------
        results : dict
            Dictionary containing the results of the simulation.

        Example
        -------
        >>> parametrization = {'eta': 0.5, 'xi': 0.3}
        >>> simulation.get_results(parametrization)
        """

        self.iterator += 1
        # Generate a unique filename on each call
        iterator = randrange(1, 1000000)
        # create job name from iteration number and model name
        self.job_name = "{}_sim_{}".format(self.model_name, iterator)

        # create runfile name from job name
        runfile_name = "{}_runfile.py".format(self.job_name)

        runfile = os.path.join(self.runfile_dir, runfile_name)

        # Generate the script which will run simulation for given params
        # (Python 2.7). The approach here is inspired by the F3DASM framework
        # available at: https://github.com/bessagroup/f3dasm

        lines = [
            'import os',
            'import run.models.%s.%s as mds' % (2 * (self.script_name, )),
            'os.chdir(r\'%s\')' % self.temp_dir,
            'model_name = \'%s\'' % self.model_name,
            'job_name = \'%s\'' % self.job_name,
            'mds.create_sim(model_name,job_name, %s, save_cae = %s)' %
            (parametrization, self.save_cae),
            'mds.post_process(job_name, %s)' % parametrization
        ]

        # Write the commands onto a runfile
        with open(runfile, 'x') as f:
            for line in lines:
                f.write(line + '\n')
            f.close()

        # Run the script using cmd
        command = f'{self.abaqus_dir} cae noGUI={runfile}'

        # Whether to suppress the ABAQUS Python output or not - setting this
        # to False might be useful for debugging
        suppress_output = True

        if suppress_output:
            FNULL = open(os.devnull, 'w')

            subprocess.call(command,
                            shell=True,
                            stdout=FNULL,
                            stderr=subprocess.STDOUT)
        else:
            os.system(command)

        # The results are stored in a pickle file
        pickle_name = self.job_name + '_results.pkl'
        results_file = os.path.join(self.temp_dir, pickle_name)

        # Unpickling the file
        with open(results_file, 'rb') as f:
            results = pickle.load(f, encoding='bytes')

        results = {k.decode('utf8'): (v, 0.0) for k, v in results.items()}

        return results

    def analytical_hex_thick(self, parametrization):
        """
        Analytical function, which is used in place of the get_results to test
        the BO framework outside of Abaqus.

        Parameters
        ----------
        parametrization : dict
            Dictionary containing parameters 'eta' and 'xi' for analysis.

        Returns
        -------
        results : dict
            Dictionary containing the results of the analysis such as
            'stress_ratio' and 'stiffness_ratio'.

        Example
        -------
        >>> parametrization = {'eta': 0.5, 'xi': 0.3}
        >>> simulation.analytical_hex_thick(parametrization)
        """
        eta = parametrization['eta']
        xi = parametrization['xi']

        stiffness_ratio = (eta**3 / (1 - xi + eta * xi)**3) * (
            1 / (xi**3 + eta**3 - eta**3 * xi**3))

        relative_density = 0.15
        # The quantities below are not normalised wrt uniform
        # plastic strength is per material yield strength - this eventually cancels out
        plastic_strength = 0.5 * relative_density**2 * 1 / (
            1 - xi + xi *
            eta)**2 if xi < eta**2 else 0.5 * relative_density**2 * eta**2 / (
                xi * (1 - xi + xi * eta)**2)
        uniform_strength = 0.5 * relative_density**2

        results = {
            'stress_ratio': (plastic_strength / uniform_strength, 0.0),
            'stiffness_ratio': (stiffness_ratio, 0.0)
        }

        return results
