

import time
import traceback
import logging
import os
import matplotlib.pyplot as plt
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
from pycqed.analysis import analysis_toolbox as a_tools

class AnalysisDaemon:
    """
    AnalysisDaemon is a class that allow to process analysis in a
    separate python kernel to allow measurements to run in parallel
    to the analysis.
    The Daemon can either be started:
     - in a separate ipython notebook using: `AnalysisDaemon(start=True)`.
       Note that the a_tools.datadir should be set before calling the daemon
       or passed with the watchdir argument
     - via the commandline by calling `analysis_daemon.py` with possible
       additional arguments (see `analysis_daemon.py --help`)
     - with the start_analysis_daemon.bat script located in pycqedscripts/scripts
       (Windows only)

    """

    def __init__(self, t_start=None, start=True, watchdir=None):
        """
        Initialize AnalysisDaemon
        Args:
            t_start (str): timestamp from which to start observing the data
                directory. If None, defaults to now.
            start (bool): whether or not to start the daemon
            watchdir (str): directory which the Daemon should look at.
                Defaults to analusis_toolbox.datadir.

        """
        self.t_start = t_start
        self.last_ts = None
        self.poll_interval = 10  # seconds
        self.errs = []
        self.job_errs = []
        if watchdir is not None:
            a_tools.datadir = watchdir
        if start:
            self.start()

    def start(self):
        """
        Starts the AnalysisDaemon
        Returns:

        """
        self.last_ts = a_tools.latest_data(older_than=self.t_start,
                                           return_path=False,
                                           return_timestamp=True, n_matches=1)[0]
        self.run()

    def run(self):
        try:
            while (True):
                self.check_job()
                for i in range(self.poll_interval):
                    time.sleep(1)
        except KeyboardInterrupt as e:
            pass
        except Exception as e:
            log.error(e)
            self.errs.append(traceback.format_exc())
            self.run()

    def check_job(self):
        """
        Checks whether new jobs have been found and processes them
        Returns:

        """
        try:
            timestamps, folders = a_tools.latest_data(newer_than=self.last_ts,
                                                      raise_exc=False,
                                                      return_timestamp=True,
                                                      n_matches=1000)
        except ValueError as e:
            return  # could not find any timestamps matching criteria
        log.info(f"Searching jobs in: {timestamps[0]} ... {timestamps[-1]}.")
        found_jobs = False
        for folder, ts in zip(folders, timestamps):
            jobs_in_folder = []
            for file in os.listdir(folder):
                if file.endswith(".job"):
                    jobs_in_folder.append(os.path.join(folder, file))
            if len(jobs_in_folder) > 0:
                log.info(f"Found {len(jobs_in_folder)} job(s) in {ts}")
                found_jobs = True

            for filename in jobs_in_folder:
                if os.path.isfile(filename):
                    time.sleep(1)  # wait to make sure that the file is fully written

                    job = self.read_job(filename)
                    errl = len(self.job_errs)
                    self.run_job(job)
                    if errl == len(self.errs):
                        os.rename(filename, filename + '.done')
                    else:
                        os.rename(filename, filename + '.failed')
                        new_errors = self.errs[errl:]
                        self.write_to_job(filename + '.failed', new_errors)
                    self.last_ts = ts
        if not found_jobs:
            log.info(f"No new job found.")

    @staticmethod
    def read_job(filename):
        job_file = open(filename, 'r')
        job = "".join(job_file.readlines())
        job_file.close()
        return job
    @staticmethod
    def write_to_job(filename, new_lines):
        job_file = open(filename, 'r+')
        job_file.write("\n")
        job_file.write("".join(new_lines))
        job_file.close()

    def run_job(self, job):
        try:
            exec(job)
            plt.close('all')
        except Exception as e:
            log.error(f"Error in job: {job}:\n{e}")
            self.job_errs.append(traceback.format_exc())

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", help="Watch directory")
    parser.add_argument("--t-start", help="Starting watch time formatted as a "
                                          "timestamp YYYYmmdd_hhMMss or string"
                                          " 'now' (default).")
    args = parser.parse_args()
    if args.data_dir is not None:
        a_tools.datadir = args.data_dir
    if args.t_start == "now":
        args.t_start = None
    ad = AnalysisDaemon(t_start=args.t_start)
