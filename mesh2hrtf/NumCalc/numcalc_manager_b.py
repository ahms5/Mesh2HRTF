"""
Run NumCalc on one or multiple Mesh2HRTF project folders.

This script monitors the RAM and CPU usage and starts a new NumCalc instance
whenever enough resources are available. A log file is written to the base
directory and an error is raised if any unfinished instances are detected.

DIFFERENCE BETWEEN NUMCALC MANAGER VERSION A AND B
--------------------------------------------------
Version A starts the frequency steps in reversed order and estimates the
required RAM after a NumCalc instance was started. The RAM usage of the next
frequency step is assumed to be the maximum ram usage of any running NumCalc
instance times value given by the parameter `ram_safety_factor`. This might
cause RAM overloads when starting NumCalc instances for low frequencies, that
often require more RAM than mid-range frequencies.

Version B estimates the required RAM for each frequency step before starting
any instance using NumCalc's `estimate_ram` option. This makes starting new
instances more flexible, because any instance that does not exceed the maximum
available RAM can be started. It might make staring new instances more secure
in some cases (see above).

HOW TO USE
----------
run ``python num_calc_manager -h`` for help (requires Python and psutil).

Tips:
  1- It is not recommended to run multiple NumCalcManager.py scripts on the
     same computer (but it would work)
  2- avoid folder names with spaces. Code is not tested for that.
  3- If there is a problem with some instance result delete its output folder
     folder "be.X" and the Manager will automatically re-run that instance on
     the next run.
"""

import os
import glob
import time
import psutil
import subprocess
import argparse
import numpy as np
import mesh2hrtf as m2h


# helping functions -----------------------------------------------------------
def raise_error(message, text_color, log_file, confirm_errors):
    """Two different ways of error handling depending on `confirm_errors`"""

    # error to logfile
    with open(log_file, "a", encoding="utf8", newline="\n") as f:
        f.write("\n\n" + message + "\n")

    # error to console
    if confirm_errors:
        print(text_color + message)
        input(text_color + "Press Enter to exit num_calc_manager\033[0m")
        raise Exception("num_calc_manager was stopped due to an error")
    else:
        raise ValueError(message)


def print_message(message, text_color, log_file):
    """Print message to console and log file"""

    print(text_color + message)

    with open(log_file, "a", encoding="utf8", newline="\n") as f:
        f.write(message + "\n")


def available_ram(ram_offset):
    """Get the available RAM = free RAM - ram_offset"""
    RAM_info = psutil.virtual_memory()
    return max([0, RAM_info.available / 1073741824 - ram_offset])


def numcalc_instances():
    """Return the number of currently running NumCalc instances"""
    num_instances = 0
    for p in psutil.process_iter(['name', 'memory_info']):
        if p.info['name'].endswith("NumCalc"):
            num_instances += 1

    return num_instances


def check_project(project, numcalc_executable, log_file):
    """
    Find unfinished instances (frequency steps) in a Mesh2HRTF project folder

    Parameters
    ----------
    project : str
        Full path of the Mesh2HRTF project folder

    Returns
    -------
    all_instances : numpy array
        Array of shape (N, 4) where N is the number of detected frequency
        steps in all source_* folders in the project. The first column contains
        the source number, the second the frequency step, the third the
        frequency in Hz, and the fourth the estimated RAM consumption in GB.
    instances_to_run : numpy array, None
        Array of size (M, 4) if any instances need to be run (in this case M
        gives the unfinished instances). ``None``, if all instances are
        finished.
    source_counter : int
        Number of sources in the project
    """

    # get source folders and number of sources
    sources = glob.glob(os.path.join(project, 'NumCalc', "source_*"))
    source_counter = len(sources)
    sources = [os.path.join(project, 'NumCalc', f"source_{s+1}")
               for s in range(source_counter)]

    # loop source_* folders
    for source_id, ff in enumerate(sources):

        # estimate RAM consumption if required
        if not os.path.isfile(os.path.join(ff, "Memory.txt")):

            print_message(f"Obtaining RAM estimates for {ff}",
                          '\033[0m', log_file)

            if os.name == 'nt':  # Windows detected
                # run NumCalc and route all printouts to a log file
                subprocess.run(
                    [f"{numcalc_executable} -estimate_ram"],
                    stdout=subprocess.DEVNULL, cwd=ff, check=True)

            else:  # elif os.name == 'posix': Linux or Mac detected
                # run NumCalc and route all printouts to a log file
                subprocess.run(
                    [f"{numcalc_executable} -estimate_ram"],
                    shell=True, stdout=subprocess.DEVNULL, cwd=ff, check=True)

        # get RAM estimates and prepend source number
        estimates = m2h.read_ram_estimates(ff)
        estimates = np.concatenate(
            ((source_id + 1) * np.ones((estimates.shape[0], 1)), estimates),
            axis=1)

        if source_id == 0:
            all_instances = estimates
            instances_to_run = None
        else:
            all_instances = np.append(all_instances, estimates, axis=0)

        # loop frequency steps
        for step in range(estimates.shape[0]):

            if not os.path.isfile(os.path.join(
                    ff, "be.out", f"be.{1 + step}", "pEvalGrid")):

                # there are no output files, process this
                if instances_to_run is None:
                    instances_to_run = np.atleast_2d(estimates[step])
                else:
                    instances_to_run = np.append(
                        instances_to_run, np.atleast_2d(estimates[step]),
                        axis=0)

            elif os.path.isfile(os.path.join(
                    ff, f'NC{1 + step}-{1 + step}.out')):

                # check if "NCx-x.out" contains "End time:" to confirm that
                # the simulation was completed.
                nc_out = os.path.join(
                    ff, f'NC{1 + step}-{1 + step}.out')
                with open(nc_out, "r", encoding="utf8", newline="\n") as f:
                    nc_out = "".join(f.readlines())

                if 'End time:' not in nc_out:
                    # instance did not finish
                    if instances_to_run is None:
                        instances_to_run = np.atleast_2d(estimates[step])
                    else:
                        instances_to_run = np.append(
                            instances_to_run, np.atleast_2d(estimates[step]),
                            axis=0)

    return all_instances, instances_to_run, source_counter


# parse command line input ----------------------------------------------------
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--project_path", default=False, type=str,
    help=("The working directory. This can be a directory that contains "
          "multiple Mesh2HRTF project folders, a Mesh2HRTF project folder or "
          "a NumCalc folder inside a Mesh2HRTF project folder. The default "
          "uses the current working directory"))
parser.add_argument(
    "--numcalc_path", default=False, type=str,
    help=("On Unix, this is the path to the NumCalc binary (by default "
          "'NumCalc' is used). On Windows, this is the path to the folder "
          "'NumCalc_WindowsExe' from "
          "https://sourceforge.net/projects/mesh2hrtf-tools/ (by default "
          "the project_path is searched for this folder)"))
parser.add_argument(
    "--wait_time", default=15, type=int,
    help=("Delay in seconds for waiting until the RAM and CPU usage is checked"
          " after a NumCalc instance was started."))
parser.add_argument(
    "--max_ram_load", default=False, type=float,
    help=("The RAM that can maximally be used in GB. New NumCalc instances are"
          " only started if enough RAM is available. By default all available "
          "RAM will be used."))
parser.add_argument(
    "--ram_safety_factor", default=1.05, type=float,
    help=("A safty factor that is applied to the estimated RAM consumption. "
          "The estimate is obtained using NumCalc -estimate_ram. The default "
          "of 1.05 would for example assume that 10.5 GB ram are needed if "
          "a RAM consumption of 10 GB was estimated by NumCalc."))
parser.add_argument(
    "--max_cpu_load", default=90, type=int,
    help="Maximum allowed CPU load in percent")
parser.add_argument(
    "--max_instances", default=0, type=int,
    help=("The maximum numbers of parallel NumCalc instances. The default of 0"
          "Launches a new instance if the current CPU load is below "
          "`max_cpu_load` and less instances than available CPU cores are "
          "currently running"))
parser.add_argument(
    "--confirm_errors", default='True', choices=('True', 'False'), type=str,
    help=("If True, num_calc_manager waits for user input in case an error "
          "occurs."))

args = vars(parser.parse_args())

# default values
args["project_path"] = args["project_path"] if args["project_path"] \
    else os.path.dirname(os.path.realpath(__file__))

if os.name == "nt":
    args["numcalc_path"] = args["numcalc_path"] if args["numcalc_path"] \
        else args["project_path"]
else:
    args["numcalc_path"] = args["numcalc_path"] if args["numcalc_path"] \
        else "NumCalc"

RAM_info = psutil.virtual_memory()
args["max_ram_load"] = RAM_info.total / 1073741824 \
    if args["max_ram_load"] is False else args["max_ram_load"]

args["max_instances"] = psutil.cpu_count() if args["max_instances"] == 0 \
    else args["max_instances"]

# write to local variables
project_path = args["project_path"]
numcalc_path = args["numcalc_path"]
seconds_to_initialize = args["wait_time"]
max_ram_load = args["max_ram_load"]
ram_safety_factor = args["ram_safety_factor"]
max_cpu_load_percent = args["max_cpu_load"]
max_instances = args["max_instances"]
confirm_errors = args["confirm_errors"] == 'True'

# RAM that should not be used
ram_offset = max([0, RAM_info.total / 1073741824 - max_ram_load])

# initialization --------------------------------------------------------------

# trick to get colored print-outs   https://stackoverflow.com/a/54955094
text_color_red = '\033[31m'
text_color_green = '\033[32m'
text_color_cyan = '\033[36m'
text_color_reset = '\033[0m'

current_time = time.strftime("%Y_%m_%d_%H-%M-%S", time.localtime())
log_file = f"numcalc_manager_{current_time}.txt"

# check input
if args["max_instances"] > psutil.cpu_count():
    raise_error(
        (f"max_instances is {max_instances} but can not be larger than "
         f"{psutil.cpu_count()} (The number of logical CPUs)"),
        text_color_red, log_file, confirm_errors)

# Detect what the project_path or "getcwd()" is pointing to:
if os.path.basename(project_path) == 'NumCalc':
    # project_path is a NumCalc folder
    all_projects = [os.path.dirname(project_path)]
    log_file = os.path.join(project_path, '..', log_file)
elif os.path.isfile(os.path.join(project_path, 'Info.txt')):
    # project_path is a Mesh2HRTF project folder
    all_projects = [project_path]
    log_file = os.path.join(project_path, log_file)
else:
    # project_path contains multiple Mesh2HRTF project folders
    all_projects = []  # list of project folders to execute
    for subdir in os.listdir(project_path):
        if os.path.isfile(os.path.join(project_path, subdir, 'Info.txt')):
            all_projects.append(os.path.join(project_path, subdir))

    log_file = os.path.join(project_path, log_file)

    # stop if no project folders were detected
    if len(all_projects) == 0:
        message = ("num_calc_manager could not detect any Mesh2HRTF projects "
                   f"at project_path={project_path}")
        raise_error(message, text_color_red, log_file, confirm_errors)

# remove old log-file
if os.path.isfile(log_file):
    os.remove(log_file)

# echo input parameters and number of Mesh2HRTF projects
current_time = time.strftime("%b %d %Y, %H:%M:%S", time.localtime())
message = ("\nStarting numcalc_manager with the following arguments "
           f"[{current_time}]\n")
message += "-" * (len(message) - 2) + "\n"
for key, value in args.items():
    message += f"{key}: {value}\n"

print_message(message, text_color_reset, log_file)
del message, key, value

# Check for NumCalc executable ------------------------------------------------
if os.name == 'nt':  # Windows detected

    # files that are needed to execute NumCalc
    NumCalc_runtime_files = ['NumCalc.exe', 'libgcc_s_seh-1.dll',
                             'libstdc++-6.dll', 'libwinpthread-1.dll']

    if os.path.isdir(os.path.join(all_projects[0], 'NumCalc_WindowsExe')):
        # located inside the project folder
        numcalc_path = os.path.join(all_projects[0], 'NumCalc_WindowsExe')
    elif os.path.isdir(os.path.join(os.path.dirname(all_projects[0]),
                                    'NumCalc_WindowsExe')):
        # located is inside the folder that contains all Mesh2HRTF projects
        numcalc_path = os.path.join(
            os.path.dirname(all_projects[0]), 'NumCalc_WindowsExe')
    elif os.path.isfile(os.path.join(all_projects[0],
                                     NumCalc_runtime_files[0])):
        # located directly in the project folder.
        numcalc_path = os.path.join(all_projects[0])
    else:
        # try path provided as it is
        pass

    # Check that each required runtime file is present:
    for calc_file in NumCalc_runtime_files:
        if not os.path.isfile(os.path.join(numcalc_path, calc_file)):
            message = (
                f"The file {calc_file} is missing or num_calc_manager could "
                f"not find the containing folder 'NumCalc_WindowsExe'")
            raise_error(message, text_color_red, log_file, confirm_errors)

    # full path to the NumCalc executable
    numcalc_executable = os.path.join(numcalc_path, "NumCalc.exe")

    del calc_file, NumCalc_runtime_files
else:
    if not numcalc_path.endswith("NumCalc"):
        raise_error("numcalc_path must end with 'NumCalc'", text_color_red,
                    log_file, confirm_errors)
    p = subprocess.Popen(f"command -v {numcalc_path}", stdout=subprocess.PIPE,
                         shell=True)
    if not len(p.stdout.read()):
        raise_error(f"NumCalc executable does not exist at {numcalc_path}",
                    text_color_red, log_file, confirm_errors)
    numcalc_executable = numcalc_path
    numcalc_path = os.path.dirname(numcalc_path)


# Check all projects that may need to be executed -----------------------------
projects_to_run = []
message = ("\nPer project summary of instances that will be run\n"
           "-------------------------------------------------\n")

message += f"Detected {len(all_projects)} Mesh2HRTF projects in\n"
message += f"{os.path.dirname(log_file)}\n\n"

for project in all_projects:
    all_instances, instances_to_run, *_ = check_project(
        project, numcalc_executable, log_file)

    if instances_to_run is not None:
        projects_to_run.append(project)
        message += (f"{len(instances_to_run)}/{len(all_instances)} frequency "
                    f"steps to run in {os.path.basename(project)}\n")
    else:
        message += f"{os.path.basename(project)} is already complete\n"

print_message(message, text_color_reset, log_file)


# loop to process all projects ------------------------------------------------
for pp, project in enumerate(projects_to_run):

    current_time = time.strftime("%b %d %Y, %H:%M:%S", time.localtime())

    # Check number of instances in project and estimate their RAM consumption
    root_NumCalc = os.path.join(project, 'NumCalc')
    all_instances, instances_to_run, source_counter = \
        check_project(project, numcalc_executable, log_file)
    total_nr_to_run = instances_to_run.shape[0]

    # Status printouts:
    message = (f"Started {os.path.basename(project)} "
               f"({pp + 1}/{len(projects_to_run)}, {current_time})")
    message = "\n" + message + "\n" + "-" * len(message) + "\n"
    if total_nr_to_run:
        message += (
            f"Running {total_nr_to_run}/{len(all_instances)} unfinished "
            "frequency steps in the project\n")
    else:
        message += (
            "All NumCalc simulations in this project are complete")
        print_message(message, text_color_reset, log_file)
        continue

    print_message(message, text_color_reset, log_file)

    # sort instances according to RAM consumption (highest first)
    instances_to_run = instances_to_run[np.argsort(instances_to_run[:, 3])]
    instances_to_run = np.flip(instances_to_run, axis=0)

    # check if available memory is enough for running the instance with the
    # highest memory consumption
    RAM_available = available_ram(ram_offset)
    if RAM_available < instances_to_run[-1, 3] * ram_safety_factor:
        raise_error((
            f"Available RAM is {round(RAM_available, 2)} GB, but frequency "
            f"step {int(instances_to_run[0, 1])} of source "
            f"{int(instances_to_run[0, 1])} requires "
            f"{round(instances_to_run[0, 3] * ram_safety_factor, 2)} GB."),
            text_color_red, log_file, confirm_errors)

    # main loop for starting instances
    started_instance = True
    while instances_to_run.shape[0]:

        # current time and resources
        current_time = time.strftime("%b %d %Y, %H:%M:%S", time.localtime())
        ram_required = np.min(instances_to_run[:, 3]) * ram_safety_factor
        ram_available = available_ram(ram_offset)
        cpu_load = psutil.cpu_percent(.1) / psutil.cpu_count()
        running_instances = numcalc_instances()

        # wait if
        # - CPU usage too high
        # - number of running instances is too large
        # - not enough RAM available
        if cpu_load > max_cpu_load_percent \
                or running_instances >= max_instances \
                or ram_available < ram_required:

            # print message (only done once between launching instances)
            if started_instance:
                print_message(
                    (f"\n... waiting for resources (checking every "
                     f"{seconds_to_initialize} seconds, {current_time}):\n"
                     f"{running_instances} NumCalc instances running\n"
                     f"{cpu_load} %CPU load\n"
                     f"{round(ram_available, 2)} GB RAM available\n"),
                    text_color_reset, log_file)
                started_instance = False

            # wait and continue
            time.sleep(seconds_to_initialize)
            continue

        # find frequency step with the highest possible RAM consumption
        for idx, ram_required in enumerate(instances_to_run[:, 3]):
            if ram_required <= ram_available:
                break

        # start new NumCalc instance
        source = int(instances_to_run[idx, 0])
        step = int(instances_to_run[idx, 1])
        progress = total_nr_to_run - instances_to_run.shape[0] + 1
        message = (
            f"{progress}/{total_nr_to_run} starting instance from: "
            f"{os.path.basename(project)} (source {source}, step {step}"
            f"({current_time})")
        print_message(message, text_color_reset, log_file)

        # change working directory
        os.chdir(os.path.join(root_NumCalc, "source_" + str(source)))

        if os.name == 'nt':  # Windows detected
            # create a log file for all print-outs
            LogFileHandle = open(f"NC{step}-{step}_log.txt", "w")
            # run NumCalc and route all printouts to a log file
            subprocess.Popen(
                f"{numcalc_executable} -istart {step} -iend {step}",
                stdout=LogFileHandle)

        else:  # elif os.name == 'posix': Linux or Mac detected
            # run NumCalc and route all printouts to a log file
            subprocess.Popen((
                f"{numcalc_executable} -istart {step} -iend {step}"
                f" >NC{step}-{step}_log.txt"), shell=True)

        started_instance = True
        instances_to_run = np.delete(instances_to_run, idx, 0)
        time.sleep(seconds_to_initialize)
    #  END of the main project loop ---

    # wait for last NumCalc instances to finish
    current_time = time.strftime("%b %d %Y, %H:%M:%S", time.localtime())
    message = (f"\n... waiting for last NumCalc instance to finish "
               f"(checking every {seconds_to_initialize} s, {current_time})")
    print_message(message, text_color_reset, log_file)
    while True:

        if numcalc_instances() == 0:
            break

        time.sleep(seconds_to_initialize)
#  END of all_projects loop ---

# Check all projects that may need to be executed -----------------------------
current_time = time.strftime("%b %d %Y, %H:%M:%S", time.localtime())

message = ("\nThe following instances did not finish\n"
           "--------------------------------------\n")

for project in all_projects:
    all_instances, instances_to_run, *_ = check_project(
        project, numcalc_executable, log_file)

    if instances_to_run is None:
        continue

    if instances_to_run.shape[0] > 0:
        message += f"{os.path.basename(project)}: "
        unfinished = [f"source {int(p[0])} step {int(p[1])}"
                      for p in projects_to_run]
        message += "; ".join(unfinished) + "\n"

if message.count("\n") > 3:
    message += f"Finished at {current_time}"
    raise_error(message, text_color_reset, log_file, confirm_errors)
else:
    message = f"\nAll NumCalc projects finished at {current_time}"
    print_message(message, text_color_reset, log_file)

    if confirm_errors:
        input(text_color_green + 'DONE. Hit Enter to exit')
        print(text_color_reset)
