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
import time
import psutil
import subprocess
import argparse


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


def get_num_calc_processes(numcalc_executable):
    """Return a list with the pid, names, and bytes of each NumCalc process"""
    pid_names_bytes = [
            (p.pid, p.info['name'], p.info['memory_info'].rss)
            for p in psutil.process_iter(['name', 'memory_info'])
            if p.info['name'] == os.path.basename(numcalc_executable)]
    return pid_names_bytes


def check_project(project):
    """
    Find unfinished instances (frequency steps) in a Mesh2HRTF project folder

    Parameters
    ----------
    project : str
        Full path of the Mesh2HRTF project folder

    Returns
    -------
    _type_
        _description_
    """

    # initialize variables
    all_instances = []           # all folder for each instance
    instances_to_run = []        # folders for each instance that must be run
    source_counter: int = 0      # but up to sources 99999 are supported
    frequency_steps_nr: int = 0  # init
    frequency_step = 0           # init
    min_frequency = 0            # init

    project_numcalc = os.path.join(project, 'NumCalc')

    # parse "Info.txt" to get info about frequencies
    with open(os.path.join(project, "Info.txt"), "r", encoding="utf8",
              newline="\n") as f:

        for line in f:
            if line.find('Minimum evaluated Frequency') != -1:
                idl = line.find(':')
                min_frequency = float(line[idl + 2:-1])
            elif line.find('Frequency Stepsize') != -1:
                idl = line.find(':')
                frequency_step = float(line[idl + 2:-1])
            elif line.find('Frequency Steps') != -1:
                idl = line.find(':')
                frequency_steps_nr = int(line[idl + 2:-1])

        del line, idl  # remove no longer needed variables

    # loop source_* folders
    for ff in os.listdir(project_numcalc):

        if not ff.startswith("source_"):
            continue

        source_counter += 1              # count this source
        source_id = int(ff[7:])

        # loop frequency steps
        for step in range(frequency_steps_nr):  # counting from zero

            # update list of all instances
            all_instances.append([source_id, 1+step])

            if not os.path.isfile(os.path.join(
                    project_numcalc, ff, "be.out", f"be.{1 + step}",
                    "pEvalGrid")):
                # there are no output files, process this
                instances_to_run.append(all_instances[-1])

            elif os.path.isfile(os.path.join(
                    project_numcalc, ff, f'NC{1 + step}-{1 + step}.out')):

                # check if "NCx-x.out" contains "End time:" to confirm that
                # the simulation was completed.
                nc_out = os.path.join(
                    project_numcalc, ff, f'NC{1 + step}-{1 + step}.out')
                with open(nc_out, "r", encoding="utf8", newline="\n") as f:
                    nc_out = "".join(f.readlines())

                if 'End time:' not in nc_out:
                    # instance did not finish
                    instances_to_run.append(all_instances[-1])

    return all_instances, instances_to_run, min_frequency, frequency_step, \
        frequency_steps_nr, source_counter


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
    help=("The RAM that can maximally be used. New NumCalc instances are only "
          "started if enough RAM is available. By default all available RAM "
          "will be used."))
parser.add_argument(
    "--ram_safety_factor", default=1.05, type=float,
    help=("A safty factor that is applied to the estimated RAM consumption. "
          "The estimate is obtained using NumCalc -estimate_ram. The dafault "
          "of 1.05 would for example assume that 10.5 GB ram are needed if "
          "a RAM consumption of 10 GB was estimated by NumCalc."))
parser.add_argument(
    "--max_cpu_load", default=80, type=int,
    help="Maximum allowed CPU load in percent")
parser.add_argument(
    "--max_instances", default=0, type=int,
    help=("The maximum numbers of parallel NumCalc instances. If this is 0 "
          "(default) the maximum number is estimated based on the CPU usage "
          "of the instance calculating the highest frequency and "
          "`max_cpu_load` (see above)"))
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

# write to local variables
project_path = args["project_path"]
numcalc_path = args["numcalc_path"]
seconds_to_initialize = args["wait_time"]
ram_safety_factor = args["ram_safety_factor"]
max_cpu_load_percent = args["max_cpu_load"]
max_instances = args["max_instances"]
confirm_errors = args["confirm_errors"] == 'True'

# initialization --------------------------------------------------------------

# trick to get colored print-outs   https://stackoverflow.com/a/54955094
text_color_red = '\033[31m'
text_color_green = '\033[32m'
text_color_cyan = '\033[36m'
text_color_reset = '\033[0m'

current_time = time.strftime("%Y_%m_%d_%H-%M-%S", time.localtime())
log_file = f"numcalc_manager_{current_time}.txt"

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
message = ("Per project summary of instances that will be run\n"
           "-------------------------------------------------\n")

message += f"Detected {len(all_projects)} Mesh2HRTF projects in\n"
message += f"{os.path.dirname(log_file)}\n\n"

for project in all_projects:
    all_instances, instances_to_run, *_ = check_project(project)

    if len(instances_to_run) > 0:
        projects_to_run.append(project)
        message += (f"{len(instances_to_run)}/{len(all_instances)} in "
                    f"{os.path.basename(project)}\n")
    else:
        message += f"{os.path.basename(project)} is already complete\n"

print_message(message, text_color_reset, log_file)
del all_projects


# loop to process all projects ------------------------------------------------
for pp, project in enumerate(projects_to_run):

    current_time = time.strftime("%b %d %Y, %H:%M:%S", time.localtime())

    # Check how many instances are in this Project:
    root_NumCalc = os.path.join(project, 'NumCalc')
    all_instances, instances_to_run, min_frequency, frequency_step, \
        frequency_steps_nr, source_counter = check_project(project)
    total_nr_to_run = len(instances_to_run)

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

    # ADVANCED sorting
    # (Build matching list of frequencies for each "instances_to_run")
    matched_freq_of_inst = [0] * len(instances_to_run)
    for inst in range(total_nr_to_run):
        idx = int(instances_to_run[inst][1])
        # frequency in Hz
        matched_freq_of_inst[inst] = int(
            min_frequency + frequency_step * (idx - 1))

    # Sort list to run the largest frequencies that consume the most RAM first
    # (needed for Both ears!!!)
    Sorting_List = sorted(zip(matched_freq_of_inst, instances_to_run),
                          reverse=True)
    # sort "instances_to_run" according to decreasing frequency
    instances_to_run = [x for _, x in Sorting_List]
    # update Matched list to correspond to "instances_to_run"
    matched_freq_of_inst = [y for y, _ in Sorting_List]
    del Sorting_List, idx

    # main loop for each instance
    for NC_ins in range(total_nr_to_run):
        # current source and frequency step
        source = instances_to_run[NC_ins][0]
        step = instances_to_run[NC_ins][1]

        # Check the RAM & run instance if feasible
        current_time = time.strftime("%b %d %Y, %H:%M:%S", time.localtime())
        RAM_info = psutil.virtual_memory()
        message = (
            f"\n{NC_ins + 1}/{total_nr_to_run} preparing "
            f"{matched_freq_of_inst[NC_ins]} Hz instance from source {source},"
            f" step {step}\n"
            f"{round((RAM_info.available / 1073741824), 2)} GB free RAM, "
            f"{RAM_info.percent}% used ({current_time})")
        print_message(message, text_color_reset, log_file)

        # use this to autodetect how many instances can at most be executed
        if NC_ins > 0 and not max_instances:
            # noinspection PyBroadException
            try:
                # it is better to get fresh pid (hopefully at least one NumCalc
                # process is still running)
                pid_names_bytes = get_num_calc_processes(numcalc_executable)

                PrcInfo = psutil.Process(pid_names_bytes[0][0])

                # wait until CPU usage is realistic
                Instance_CPU_usageNow = 0
                while Instance_CPU_usageNow < 0.001:
                    Instance_CPU_usageNow = \
                        PrcInfo.cpu_percent() / psutil.cpu_count()
                    time.sleep(0.01)

                # calculate optimal maximum number of NumCalc processes
                max_instances = round(
                    max_cpu_load_percent / Instance_CPU_usageNow)

                message = (
                    "One NumCalc instance requires "
                    f"{round(Instance_CPU_usageNow, 1)}% of the CPU. "
                    "max_instances is now automatically set to "
                    f"{max_instances}")
                print_message(message, text_color_reset, log_file)

            except BaseException:
                message = (
                    "Failed to automatically set the maximum number of "
                    "parallel NumCalc instances. This can happen if a NumCalc "
                    "process finished very fast. Try to lower wait_time or "
                    "manually set max_instances")
                raise_error(message, text_color_red, log_file, confirm_errors)

        #  Main checks before launching the next instance
        # (to avoid system resource overload)
        wait_for_resources = False if NC_ins == 0 else True

        while wait_for_resources:
            # Start time, number of numcalc processes and RAM usage
            current_time = \
                    time.strftime("%b %d %Y, %H:%M:%S", time.localtime())
            pid_names_bytes = get_num_calc_processes(numcalc_executable)
            RAM_info = psutil.virtual_memory()

            # DEBUGGING --- Find Processes consuming more than 250MB of memory:
            # pid_names_bytes = [
            #     (p.pid, p.info['name'], p.info['memory_info'].rss)
            #     for p in psutil.process_iter(['name', 'memory_info'])
            #     if p.info['memory_info'].rss > 250 * 1048576]

            # start NumCalc instance if none is running
            if len(pid_names_bytes) == 0:
                max_numcalc_ram = 0
                break

            # if the maximum number of instances to launch is not exceeded
            elif len(pid_names_bytes) < max_instances:

                # find out how much RAM is consumed by any NumCalc Instance
                max_numcalc_ram = pid_names_bytes[0][2]
                if len(pid_names_bytes) > 1:
                    for prcNr in range(1, len(pid_names_bytes)):
                        if pid_names_bytes[prcNr][2] > max_numcalc_ram:
                            max_numcalc_ram = pid_names_bytes[prcNr][2]

                # check if we can run more:
                # IF free RAM is greater than RAM consumption of the biggest
                # NumCalc instance x ram_safety_factor
                if RAM_info.available > max_numcalc_ram * ram_safety_factor:
                    message = (
                        "... Starting next instance: "
                        f"{round((RAM_info.available / 1073741824), 1)} GB "
                        f"free RAM detected [{current_time}]")
                    print_message(message, text_color_reset, log_file)
                    break

                else:
                    if not message.startswith("... Waiting for free RAM"):
                        message = (
                            "... Waiting for free RAM:"
                            f"{round((RAM_info.available / 1073741824), 1)} "
                            f"GB free RAM detected. "
                            f"{round((max_numcalc_ram * ram_safety_factor / 1073741824), 1)} "  # noqa
                            f"GB needed [{current_time}, checking every "
                            f"{seconds_to_initialize} seconds]")
                        print_message(message, text_color_reset, log_file)

            else:
                if not message.startswith(
                        f"... Waiting for 1/{max_instances}"):
                    message = (
                        f"... Waiting for 1/{max_instances} instances to "
                        "finish: "
                        f"{round((RAM_info.available / 1073741824), 1)} GB "
                        f"free RAM detected [{current_time}, checking every "
                        f"{seconds_to_initialize} seconds]")
                    print_message(message, text_color_reset, log_file)

            # delay before trying the while loop again
            time.sleep(seconds_to_initialize)
            # END of the while loop ---

        # start next instance -------------------------------------------------
        message = (
            f"{NC_ins + 1}/{total_nr_to_run} starting instance from: "
            f"{os.path.basename(project)} (source {source}, step {step}")
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

        # optimize waiting time (important if available RAM >>> needed RAM)
        if NC_ins > 0:
            # noinspection PyUnboundLocalVariable
            if RAM_info.available > max_numcalc_ram * 3:
                # wait less, if RAM is enough for three new instances
                waitTime = 0.5
            elif RAM_info.available > max_numcalc_ram * 2:
                # wait less if RAM is enough for two new instances
                waitTime = seconds_to_initialize / 2
            else:
                # wait longer to assess how much RAM is available
                waitTime = seconds_to_initialize

        else:
            # always wait for the 1st instance to initialize to get worst-case
            # RAM use estimate
            waitTime = seconds_to_initialize

        # Wait for instance to initialize before attempting to start the next
        message = f"... waiting {waitTime} s for instance to initialize RAM"
        print_message(message, text_color_reset, log_file)
        time.sleep(waitTime)
    #  END of the main project loop ---

    # wait for last NumCalc instances to finish
    while True:
        # Find all NumCalc Processes
        pid_names_bytes = get_num_calc_processes(numcalc_executable)

        # no NumCalc processes are running, so Finish
        if len(pid_names_bytes) == 0:
            break

        message = (f"... waiting {2 * seconds_to_initialize} s for last "
                   "processes to finish")
        print_message(message, text_color_reset, log_file)
        time.sleep(2 * seconds_to_initialize)
#  END of all_projects loop ---

# Check all projects that may need to be executed -----------------------------
current_time = time.strftime("%b %d %Y, %H:%M:%S", time.localtime())

projects_to_run = []
message = ("\nThe following instances did not finish\n"
           "--------------------------------------\n")

for project in projects_to_run:
    all_instances, instances_to_run, *_ = check_project(project)

    if len(instances_to_run) > 0:
        projects_to_run.append(project)
        message += f"{os.path.basename(project)}: "
        unfinished = [f"source {p[0]} step {p[1]}" for p in projects_to_run]
        message += "; ".join(unfinished) + "\n"

if message.count("\n") > 3:
    message += f"Finished at {current_time}"
    raise_error(message, text_color_reset, log_file, confirm_errors)
else:
    message = f"All NumCalc projects finished at {current_time}"
    print_message(message, text_color_reset, log_file)

    if confirm_errors:
        input(text_color_green + 'DONE. Hit Enter to exit')
        print(text_color_reset)
