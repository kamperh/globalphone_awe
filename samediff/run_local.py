#!/usr/bin/env python

"""
Run the same command in parallel on an SGE grid.

As an example, run::

    ./run_local.py 1 3 log.JOB "echo start;sleep 10;echo finished job JOB"

The final line of output is the last spawned PID.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2014, 2018, 2019
"""

import argparse
import subprocess
import sys
import re

shell = lambda command: subprocess.Popen(
    command, shell=True, stdout=subprocess.PIPE
    ).communicate()[0]


def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__.strip().split("\n")[0], add_help=False
        )
    parser.add_argument("JOB_start", type=int, help="JOB id start value")
    parser.add_argument(
        "JOB_end", type=int, help="JOB id end value (exclusive)"
        )
    parser.add_argument(
        "log_fn", type=str,
        help="log file, substituting JOB for the current id"
        )
    parser.add_argument(
        "command", type=str,
        help="execute this command, substituting JOB for the current" 
        " id (enclose in quotes if using parameters)"
        )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def main():
    args = check_argv()
    job_start = args.JOB_start
    job_end = args.JOB_end
    log_fn = args.log_fn
    command = args.command

    pid = -1
    for i in range(job_start, job_end + 1):
        cur_command = re.sub("JOB", str(i), command)
        cur_log = re.sub("JOB", str(i), log_fn)
        pid = subprocess.Popen(
            cur_command, shell=True, stderr=subprocess.STDOUT,
            stdout=open(cur_log, "wb")
            ).pid
        print("Spawning job " + str(i) + " with PID:", pid)
    print(pid)


if __name__ == "__main__":
    main()
