#!/bin/bash

set -x

echo "args: $@"

# set the dataset dir
[[ -z $DATADIR ]] && DATADIR='/eos/cms/store/group/ml/Tagging4ScoutingHackathon/Adelina/hbb/ak8'

# set the dataset dir
[[ -z $OUTPUT ]] && OUTPUT='/eos/cms/store/group/ml/Tagging4ScoutingHackathon/Adelina/hbb/ak8/training_output/Run3Summer22'

# set a comment via `COMMENT`
suffix=${COMMENT}

# set the number of gpus for DDP training via `DDP_NGPUS`
NGPUS=${DDP_NGPUS}
[[ -z $NGPUS ]] && NGPUS=1
if ((NGPUS > 1)); then
    CMD="torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS $(which weaver) --backend nccl"
else
    CMD="weaver"
fi

data_config="data/massreg.yaml"
model_config="networks/massreg.py"

epochs=50
samples_per_epoch=$((10000 * 1024 / $NGPUS))
samples_per_epoch_val=$((10000 * 128))
dataopts="--num-workers 4 --fetch-step 0.01"
batchopts="--batch-size 1024 --start-lr 1e-2"

$CMD \
    --regression-mode \
    --demo \
    --data-train \
    "BulkGravitonToHHTo4Q:${DATADIR}/BulkGravitonToHH_MX*/Run3Summer22/*/*/hadd.root" \
    "QCD:${DATADIR}/QCD_PT-*/Run3Summer22/*/*/hadd.root" \
    --data-config ${data_config} --network-config ${model_config} \
    --model-prefix ${OUTPUT}/ak8_massreg_{auto}${suffix}/net \
    $dataopts $batchopts \
    --samples-per-epoch ${samples_per_epoch} --samples-per-epoch-val ${samples_per_epoch_val} --num-epochs $epochs --gpus "" \
    --optimizer ranger --log ${OUTPUT}/ak8_massreg_{auto}${suffix}.log --predict-output pred.root \
    "${@:2}"
