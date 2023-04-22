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
    """"Class for storing simulation data and running trials in ABAQUS """

    def __init__(self, model_name, script_name, result_metrics, temp_dir, runfile_dir, abaqus_dir='abaqus',
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

    def clean_up_prompt(self):

        if yes_or_no('Running the program will delete all previous simulations. Proceed?'):
            clean_directory(self.temp_dir)
            clean_directory(self.runfile_dir)
        else:
            print('Terminating the script')
            exit()

    def get_results(self, parametrization):
        self.iterator += 1
        # unique filename generated on each call
        iterator = randrange(1, 1000000)
        # create job name from iteration number and model name
        self.job_name = "{}_sim_{}".format(self.model_name, iterator)

        # create runfile name from job name
        runfile_name = "{}_runfile.py".format(self.job_name)

        runfile = os.path.join(self.runfile_dir, runfile_name)

        # Generate the script which will run simulation for given params (Python 2.7)

        lines = ['import os',
                 'import run.models.%s.%s as mds' % (2 * (self.script_name,)),
                 'os.chdir(r\'%s\')' % self.temp_dir,
                 'model_name = \'%s\'' % self.model_name,
                 'job_name = \'%s\'' % self.job_name,
                 'mds.create_sim(model_name,job_name, %s, save_cae = %s)' % (parametrization, self.save_cae),
                 'mds.post_process(job_name, %s)' % parametrization]

        # write the commands onto a runfile
        with open(runfile, 'x') as f:
            for line in lines:
                f.write(line + '\n')
            f.close()

        # Run the script using cmd                    
        command = f'{self.abaqus_dir} cae noGUI={runfile}'

        # Normally set to True but False is good for debuging
        suppress_output = False

        if suppress_output:
            FNULL = open(os.devnull, 'w')

            subprocess.call(command,
                            shell=True,
                            stdout=FNULL,
                            stderr=subprocess.STDOUT)
        else:
            os.system(command)

        # result file is stored within temp so it must be extracted
        pickle_name = self.job_name + '_results.pkl'
        results_file = os.path.join(self.temp_dir, pickle_name)

        # Unpickling the file 
        with open(results_file, 'rb') as f:
            results = pickle.load(f, encoding='bytes')

        results = {k.decode('utf8'): (v, 0.0) for k, v in results.items()}

        return results

    def analytical_hex_thick(self, parametrization):
        eta = parametrization['eta']
        xi = parametrization['xi']

        stiffness_ratio = (eta ** 3 / (1 - xi + eta * xi) ** 3) * (1 / (xi ** 3 + eta ** 3 - eta ** 3 * xi ** 3))

        relative_density = 0.15
        # The quantities below are not normalised wrt uniform
        # plastic strength is per material yield strength - this eventually cancels out
        plastic_strength = 0.5 * relative_density ** 2 * 1 / (
                    1 - xi + xi * eta) ** 2 if xi < eta ** 2 else 0.5 * relative_density ** 2 * eta ** 2 / (
                    xi * (1 - xi + xi * eta) ** 2)
        uniform_strength = 0.5 * relative_density ** 2

        results = {'stress_ratio': (plastic_strength / uniform_strength,0.0),
                   'stiffness_ratio': (stiffness_ratio, 0.0)}

        return results
