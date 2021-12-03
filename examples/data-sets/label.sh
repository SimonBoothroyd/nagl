#!/bin/bash
#
# Set the job name and wall time limit
#BSUB -J nagl
#BSUB -W 168:00
#
# Set the output and error output paths.
#BSUB -o  %J.o
#BSUB -e  %J.e
#
# Set any cpu options.
#BSUB -n 1 -R "span[ptile=1]"
#BSUB -M 16

# Enable conda
. ~/.bashrc

# Use the right conda environment
conda activate nagl

rm -rf labelled && mkdir labelled

# Compute the AM1 partial charges and multi-conformer WBO for each molecule.
for name in "enamine-10240.sdf.gz" \
            "enamine-50240.sdf.gz" \
            "NCI-Open_2012-05-01.sdf.gz" \
            "ChEMBL_eps_78.sdf.gz" \
            "ZINC_eps_78.sdf.gz" \
            "OpenFF-Industry-Benchmark-Season-1-v1-1.smi"
do

  nagl label --input "processed/${name}"            \
             --output "labelled/${name%%.*}.sqlite" \
             --n-workers 250                        \
             --batch-size 250                       \
             --worker-type lsf                      \
             --lsf-memory 4                         \
             --lsf-walltime "32:00"                 \
             --lsf-queue "cpuqueue"                 \
             --lsf-env "nagl"

done
