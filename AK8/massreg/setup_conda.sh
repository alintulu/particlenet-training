#!/bin/bash

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/afs/cern.ch/work/a/adlintul/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/afs/cern.ch/work/a/adlintul/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/afs/cern.ch/work/a/adlintul/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/afs/cern.ch/work/a/adlintul/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
conda activate weaver

echo "Setting up CUDA..."
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

#nvcc --version
#nvidia-smi
export TMPDIR=`pwd`

