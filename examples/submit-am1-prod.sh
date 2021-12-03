#!/bin/bash
#
# Set the job name and wall time limit
#BSUB -J am1
#BSUB -W 02:00
#
# Set the output and error output paths.
#BSUB -o  %J.o
#BSUB -e  %J.e
#
# Set any gpu options.
#BSUB -q gpuqueue
#BSUB -gpu num=1:j_exclusive=yes:mode=shared:mps=no:
#
#BSUB -M 5

# Enable conda
. ~/.bashrc

conda activate nagl
conda env export > conda-env.yml

# Launch my program.
module load cuda/11.0

python train-am1-q-model.py --train-set             "data-sets/labelled/ChEMBL_eps_78.sqlite" \
                            --train-set             "data-sets/labelled/ZINC_eps_78.sqlite"   \
                            --train-batch-size      256                                       \
                            --val-set               "data-sets/labelled/enamine-10240.sqlite" \
                            --test-set              "data-sets/labelled/OpenFF-Industry-Benchmark-Season-1-v1-1.sqlite" \
                            --n-gcn-layers          5                                         \
                            --n-gcn-hidden-features 128                                       \
                            --n-am1-layers          2                                         \
                            --n-am1-hidden-features 64                                        \
                            --learning-rate         0.001                                     \
                            --n-epochs              400
