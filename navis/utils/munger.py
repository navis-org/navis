#!/usr/bin/env python3
# munger.py
# script to process all of the images in the specified directory tree
# 1) Run affine transform
# 2) Run Warp transform
# 3) Reformat

import argparse
import os
import sys
import signal
import gzip
from pathlib import Path

version = "1.21"
is_win = os.name == 'nt'

# Autoflush stdout
sys.stdout.reconfigure(line_buffering=True)

# Global variable to handle QUIT signal
quit_next_image = False

def interrupt(signal, frame):
    global quit_next_image
    quit_next_image = True

signal.signal(signal.SIGQUIT, interrupt)

def init():
    parser = argparse.ArgumentParser(description='Process images in a directory tree.')
    parser.add_argument('-E', type=str, default="1e-1", help='Energy weight')
    parser.add_argument('-X', type=str, default="16", help='Exploration')
    parser.add_argument('-M', type=str, default="nmi", help='Metric')
    parser.add_argument('-C', type=str, default="4", help='Coarsest')
    parser.add_argument('-G', type=str, default="40", help='Grid spacing')
    parser.add_argument('-R', type=str, default="3", help='Refine')
    parser.add_argument('-J', type=str, default="0", help='Jacobian')
    parser.add_argument('-o', type=str, default="nrrd", help='Output type')
    parser.add_argument('-b', type=str, help='Binary directory')
    parser.add_argument('-s', type=str, required=True, help='Reference image')
    parser.add_argument('-v', action='store_true', help='Verbose output')
    parser.add_argument('-T', type=str, default="auto", help='Threads')
    parser.add_argument('-c', type=str, default="01", help='Registration channels')
    parser.add_argument('-r', type=str, default="01", help='Reformat channels')
    parser.add_argument('-l', type=str, default="f", help='Reformat levels')
    parser.add_argument('-f', type=str, default="01", help='Reference image channel')
    parser.add_argument('-d', type=str, help='Output directory')
    parser.add_argument('-k', type=str, help='Lock message')
    parser.add_argument('-x', type=str, default="never", help='Delete input image')
    parser.add_argument('-m', type=int, default=8760, help='Max time in hours')
    parser.add_argument('-p', type=str, help='Script file')
    parser.add_argument('input_files', nargs='+', help='Input files or directories')
    return parser.parse_args()

def arrayidx(value, options):
    try:
        return options.index(value)
    except ValueError:
        return -1

def main():
    args = init()

    energyweight = args.E
    exploration = args.X
    metric = args.M
    coarsest = args.C
    gridspacing = args.G
    refine = args.R
    jacobian = args.J
    output_type = args.o

    metric_options = ["nmi", "mi", "cr", "msd", "ncc"]
    metric_index = arrayidx(metric, metric_options)
    if metric_index == -1:
        sys.exit(f"Unrecognized metric {metric}")

    warp_suffix = f"warp_m{metric_index}g{gridspacing}c{coarsest}e{energyweight}x{exploration}r{refine}"
    icweight = args.I if args.I else "0"

    hostname = os.uname().nodename
    print(f"hostname = {hostname}")

    if is_win:
        if args.v:
            os.system('echo User path is %PATH%')
    else:
        if args.v:
            os.system('echo User path is $PATH')

    threads = args.T
    reference_image = args.s
    if args.v:
        print(f"Reference brain is {reference_image}")

    if not os.path.isfile(reference_image):
        sys.exit(f"Unable to read reference brain {reference_image}")

    reference_stem = Path(reference_image).stem
    reference_stem = reference_stem.replace('_warp', '-warp').replace('_9dof', '-9dof')
    reference_stem = reference_stem.split('.')[0].split('_')[0]
    if args.v:
        print(f"Reference brain stem is {reference_stem}")

    bin_dir = args.b if args.b else "@CMTK_BINARY_DIR_CONFIG@"
    if not os.path.isdir(bin_dir):
        sys.exit(f"Can't access binary directory {bin_dir}")

    warp_command = os.path.join(bin_dir, "warp")
    aff_command = os.path.join(bin_dir, "registration")
    initial_aff_command = os.path.join(bin_dir, "make_initial_affine")
    landmarks_aff_command = os.path.join(bin_dir, "align_landmarks")
    reformat_command = os.path.join(bin_dir, "reformatx")

    reg_channels = args.c
    reformat_channels = args.r
    reformat_levels = args.l
    reference_image_channel = args.f

    reg_root = "Registration"
    image_root = "images"
    reformat_root = "reformatted"

    if args.d:
        dir = args.d
        if dir.startswith('.'):
            reg_root += args.d
            reformat_root += args.d
        else:
            reg_root = args.d

    lockmessage = args.k if args.k else f"{hostname}:{os.getpgrp()}" if not is_win else ""
    print(f"JOB ID = {lockmessage}")

    delete_input_image = args.x
    affine_total = 0
    initial_affine_total = 0
    affine_total_failed = 0
    initial_affine_total_failed = 0
    warp_total = 0
    reformat_total = 0
    maxtime = args.m * 3600
    starttime = time.time()
    if args.v:
        print(f"Start time is: {starttime} seconds")

    found = {}
    if args.p:
        if args.v:
            print(f"Generating script file {args.p}")
        try:
            script_file = open(args.p, 'w')
        except IOError:
            sys.exit(f"Unable to open script file {args.p} for writing")

    root_dir = os.getcwd()
    if is_win:
        root_dir = os.path.abspath(os.sep)
    print(f"Root directory is {root_dir}")

    nargs = len(args.input_files)
    if nargs < 1:
        sys.exit("Usage: munger.py [options] input_files")

    if args.v:
        print(f"There are {nargs} arguments")

    for input_file_spec in args.input_files:
        # Process each input file or directory
        pass

if __name__ == "__main__":
    main()