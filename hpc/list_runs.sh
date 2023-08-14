#!/bin/bash

# Get the directory to search
directory="${VSC_SCRATCH}/wandb"

# Find all files and folders that start with 'run'
find $directory -type d -name 'offline-run*' > finished_logs.txt