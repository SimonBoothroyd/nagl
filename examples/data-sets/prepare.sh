#!/bin/bash
#
# Set the job name and wall time limit
#BSUB -J nagl
#BSUB -W 24:00
#
# Set the output and error output paths.
#BSUB -o  %J.o
#BSUB -e  %J.e
#
# Set any cpu options.
#BSUB -n 20 -R "span[ptile=20]"
#BSUB -M 2

# Enable conda
. ~/.bashrc

# Use the right conda environment
conda activate nagl

# Filter the NCI and Enamine sets according to the criteria proposed by
# Bleiziffer, Schaller and Riniker (see 10.1021/acs.jcim.7b00663)
rm -rf processed && mkdir processed

for name in "enamine-10240" "enamine-50240" "NCI-Open_2012-05-01"
do

  nagl prepare filter --input "raw/${name}.sdf.gz" \
                      --output "processed/${name}.sdf.gz" \
                      --strip-ions \
                      --n-processes 20

done

# We don't need to filter the Rinker sets are they are provided in their
# processed form or the OpenFF data set as this was curated by hand.
for name in "ChEMBL_eps_78.sdf.gz" \
            "ZINC_eps_78.sdf.gz" \
            "OpenFF-Industry-Benchmark-Season-1-v1-1.smi"
do

  cp "raw/${name}" "processed/${name}"

done
