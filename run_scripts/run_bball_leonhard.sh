# Make sure to replace this with the directory containing the data files
DATA_PATH='data/bball/'

BASE_RESULTS_DIR="results/bball/"
NUM_CPU_CORES=1
MEM_PER_CPU_CORE=8192
RUN_TIME='120:00'
NUM_GPU_CORES=8

for SEED in {1..5}
do
    ########################    DFNRI
    ###### DFNRI – 2, 2, skip_first
    WORKING_DIR="${BASE_RESULTS_DIR}dfnri_2_2_edge_skip_first/seed_${SEED}/"
    ENCODER_ARGS="--encoder_hidden 256 --encoder_mlp_num_layers 3 --encoder_mlp_hidden 128 --encoder_rnn_hidden 64"
    DECODER_ARGS="--decoder_hidden 256"
    HIDDEN_ARGS="--rnn_hidden 64"
    PRIOR_ARGS="--use_learned_prior --prior_num_layers 3 --prior_hidden_size 128"
    MODEL_ARGS="--model_type dfnri --graph_type dynamic --skip_first --layer_num_edge_types 2 2 $ENCODER_ARGS $DECODER_ARGS $HIDDEN_ARGS $PRIOR_ARGS --seed ${SEED}"
    TRAINING_ARGS='--batch_size 56 --lr 5e-4 --use_adam --num_epochs 100 --normalize_kl --normalize_nll --tune_on_nll --val_teacher_forcing --teacher_forcing_steps -1'
    mkdir -p $WORKING_DIR
    bsub -J "bball" -n $NUM_CPU_CORES -W $RUN_TIME -R "rusage[mem=$MEM_PER_CPU_CORE,ngpus_excl_p=$NUM_GPU_CORES]" -R "select[gpu_mtotal0>=10240]" <<ENDBSUB
    python -u dnri/experiments/bball_experiment.py --gpu --mode train --data_path $DATA_PATH --working_dir $WORKING_DIR $MODEL_ARGS $TRAINING_ARGS |& tee "${WORKING_DIR}results.txt"
    python -u dnri/experiments/bball_experiment.py --gpu --mode eval --load_best_model --data_path $DATA_PATH --working_dir $WORKING_DIR $MODEL_ARGS $TRAINING_ARGS |& tee "${WORKING_DIR}eval_results.txt"    
ENDBSUB
    ###### DFNRI – 2, 2
    WORKING_DIR="${BASE_RESULTS_DIR}dfnri_2_2_edge/seed_${SEED}/"
    ENCODER_ARGS="--encoder_hidden 256 --encoder_mlp_num_layers 3 --encoder_mlp_hidden 128 --encoder_rnn_hidden 64"
    DECODER_ARGS="--decoder_hidden 256"
    HIDDEN_ARGS="--rnn_hidden 64"
    PRIOR_ARGS="--use_learned_prior --prior_num_layers 3 --prior_hidden_size 128"
    MODEL_ARGS="--model_type dfnri --graph_type dynamic --layer_num_edge_types 2 2 $ENCODER_ARGS $DECODER_ARGS $HIDDEN_ARGS $PRIOR_ARGS --seed ${SEED}"
    TRAINING_ARGS='--batch_size 56 --lr 5e-4 --use_adam --num_epochs 100 --normalize_kl --normalize_nll --tune_on_nll --val_teacher_forcing --teacher_forcing_steps -1'
    mkdir -p $WORKING_DIR
    bsub -J "bball" -n $NUM_CPU_CORES -W $RUN_TIME -R "rusage[mem=$MEM_PER_CPU_CORE,ngpus_excl_p=$NUM_GPU_CORES]" -R "select[gpu_mtotal0>=10240]" <<ENDBSUB
    python -u dnri/experiments/bball_experiment.py --gpu --mode train --data_path $DATA_PATH --working_dir $WORKING_DIR $MODEL_ARGS $TRAINING_ARGS |& tee "${WORKING_DIR}results.txt"
    python -u dnri/experiments/bball_experiment.py --gpu --mode eval --load_best_model --data_path $DATA_PATH --working_dir $WORKING_DIR $MODEL_ARGS $TRAINING_ARGS |& tee "${WORKING_DIR}eval_results.txt"
ENDBSUB
    ###### DFNRI – 2, 2, 2, skip_first
    WORKING_DIR="${BASE_RESULTS_DIR}dfnri_2_2_2_edge_skip_first/seed_${SEED}/"
    ENCODER_ARGS="--encoder_hidden 256 --encoder_mlp_num_layers 3 --encoder_mlp_hidden 128 --encoder_rnn_hidden 64"
    DECODER_ARGS="--decoder_hidden 256"
    HIDDEN_ARGS="--rnn_hidden 64"
    PRIOR_ARGS="--use_learned_prior --prior_num_layers 3 --prior_hidden_size 128"
    MODEL_ARGS="--model_type dfnri --graph_type dynamic --skip_first --layer_num_edge_types 2 2 2 $ENCODER_ARGS $DECODER_ARGS $HIDDEN_ARGS $PRIOR_ARGS --seed ${SEED}"
    TRAINING_ARGS='--batch_size 56 --lr 5e-4 --use_adam --num_epochs 100 --normalize_kl --normalize_nll --tune_on_nll --val_teacher_forcing --teacher_forcing_steps -1'
    mkdir -p $WORKING_DIR
    bsub -J "bball" -n $NUM_CPU_CORES -W $RUN_TIME -R "rusage[mem=$MEM_PER_CPU_CORE,ngpus_excl_p=$NUM_GPU_CORES]" -R "select[gpu_mtotal0>=10240]" <<ENDBSUB
    python -u dnri/experiments/bball_experiment.py --gpu --mode train --data_path $DATA_PATH --working_dir $WORKING_DIR $MODEL_ARGS $TRAINING_ARGS |& tee "${WORKING_DIR}results.txt"
    python -u dnri/experiments/bball_experiment.py --gpu --mode eval --load_best_model --data_path $DATA_PATH --working_dir $WORKING_DIR $MODEL_ARGS $TRAINING_ARGS |& tee "${WORKING_DIR}eval_results.txt"
ENDBSUB
    ###### DFNRI – 2, 2, 2
    WORKING_DIR="${BASE_RESULTS_DIR}dfnri_2_2_2_edge/seed_${SEED}/"
    ENCODER_ARGS="--encoder_hidden 256 --encoder_mlp_num_layers 3 --encoder_mlp_hidden 128 --encoder_rnn_hidden 64"
    DECODER_ARGS="--decoder_hidden 256"
    HIDDEN_ARGS="--rnn_hidden 64"
    PRIOR_ARGS="--use_learned_prior --prior_num_layers 3 --prior_hidden_size 128"
    MODEL_ARGS="--model_type dfnri --graph_type dynamic --layer_num_edge_types 2 2 2 $ENCODER_ARGS $DECODER_ARGS $HIDDEN_ARGS $PRIOR_ARGS --seed ${SEED}"
    TRAINING_ARGS='--batch_size 56 --lr 5e-4 --use_adam --num_epochs 100 --normalize_kl --normalize_nll --tune_on_nll --val_teacher_forcing --teacher_forcing_steps -1'
    mkdir -p $WORKING_DIR
    bsub -J "bball" -n $NUM_CPU_CORES -W $RUN_TIME -R "rusage[mem=$MEM_PER_CPU_CORE,ngpus_excl_p=$NUM_GPU_CORES]" -R "select[gpu_mtotal0>=10240]" <<ENDBSUB
    python -u dnri/experiments/bball_experiment.py --gpu --mode train --data_path $DATA_PATH --working_dir $WORKING_DIR $MODEL_ARGS $TRAINING_ARGS |& tee "${WORKING_DIR}results.txt"
    python -u dnri/experiments/bball_experiment.py --gpu --mode eval --load_best_model --data_path $DATA_PATH --working_dir $WORKING_DIR $MODEL_ARGS $TRAINING_ARGS |& tee "${WORKING_DIR}eval_results.txt"
ENDBSUB
    ###### DFNRI – 2, 2, 2, 2, skip_first
    WORKING_DIR="${BASE_RESULTS_DIR}dfnri_2_2_2_2_edge_skip_first/seed_${SEED}/"
    ENCODER_ARGS="--encoder_hidden 256 --encoder_mlp_num_layers 3 --encoder_mlp_hidden 128 --encoder_rnn_hidden 64"
    DECODER_ARGS="--decoder_hidden 256"
    HIDDEN_ARGS="--rnn_hidden 64"
    PRIOR_ARGS="--use_learned_prior --prior_num_layers 3 --prior_hidden_size 128"
    MODEL_ARGS="--model_type dfnri --graph_type dynamic --skip_first --layer_num_edge_types 2 2 2 2 $ENCODER_ARGS $DECODER_ARGS $HIDDEN_ARGS $PRIOR_ARGS --seed ${SEED}"
    TRAINING_ARGS='--batch_size 56 --lr 5e-4 --use_adam --num_epochs 100 --normalize_kl --normalize_nll --tune_on_nll --val_teacher_forcing --teacher_forcing_steps -1'
    mkdir -p $WORKING_DIR
    bsub -J "bball" -n $NUM_CPU_CORES -W $RUN_TIME -R "rusage[mem=$MEM_PER_CPU_CORE,ngpus_excl_p=$NUM_GPU_CORES]" -R "select[gpu_mtotal0>=10240]" <<ENDBSUB
    python -u dnri/experiments/bball_experiment.py --gpu --mode train --data_path $DATA_PATH --working_dir $WORKING_DIR $MODEL_ARGS $TRAINING_ARGS |& tee "${WORKING_DIR}results.txt"
    python -u dnri/experiments/bball_experiment.py --gpu --mode eval --load_best_model --data_path $DATA_PATH --working_dir $WORKING_DIR $MODEL_ARGS $TRAINING_ARGS |& tee "${WORKING_DIR}eval_results.txt"
ENDBSUB
    ###### DFNRI – 2, 2, 2, 2
    WORKING_DIR="${BASE_RESULTS_DIR}dfnri_2_2_2_2_edge/seed_${SEED}/"
    ENCODER_ARGS="--encoder_hidden 256 --encoder_mlp_num_layers 3 --encoder_mlp_hidden 128 --encoder_rnn_hidden 64"
    DECODER_ARGS="--decoder_hidden 256"
    HIDDEN_ARGS="--rnn_hidden 64"
    PRIOR_ARGS="--use_learned_prior --prior_num_layers 3 --prior_hidden_size 128"
    MODEL_ARGS="--model_type dfnri --graph_type dynamic --layer_num_edge_types 2 2 2 2 $ENCODER_ARGS $DECODER_ARGS $HIDDEN_ARGS $PRIOR_ARGS --seed ${SEED}"
    TRAINING_ARGS='--batch_size 56 --lr 5e-4 --use_adam --num_epochs 100 --normalize_kl --normalize_nll --tune_on_nll --val_teacher_forcing --teacher_forcing_steps -1'
    mkdir -p $WORKING_DIR
    bsub -J "bball" -n $NUM_CPU_CORES -W $RUN_TIME -R "rusage[mem=$MEM_PER_CPU_CORE,ngpus_excl_p=$NUM_GPU_CORES]" -R "select[gpu_mtotal0>=10240]" <<ENDBSUB
    python -u dnri/experiments/bball_experiment.py --gpu --mode train --data_path $DATA_PATH --working_dir $WORKING_DIR $MODEL_ARGS $TRAINING_ARGS |& tee "${WORKING_DIR}results.txt"
    python -u dnri/experiments/bball_experiment.py --gpu --mode eval --load_best_model --data_path $DATA_PATH --working_dir $WORKING_DIR $MODEL_ARGS $TRAINING_ARGS |& tee "${WORKING_DIR}eval_results.txt"
ENDBSUB

    ########################    DNRI
    ###### DNRI – 4, skip_first
    WORKING_DIR="${BASE_RESULTS_DIR}dnri_4_edge_skip_first/seed_${SEED}/"
    ENCODER_ARGS="--encoder_hidden 256 --encoder_mlp_num_layers 3 --encoder_mlp_hidden 128 --encoder_rnn_hidden 64"
    DECODER_ARGS="--decoder_hidden 256"
    HIDDEN_ARGS="--rnn_hidden 64"
    PRIOR_ARGS="--use_learned_prior --prior_num_layers 3 --prior_hidden_size 128"
    MODEL_ARGS="--model_type dnri --graph_type dynamic --skip_first --num_edge_types 4 $ENCODER_ARGS $DECODER_ARGS $HIDDEN_ARGS $PRIOR_ARGS --seed ${SEED}"
    TRAINING_ARGS='--batch_size 56 --lr 5e-4 --use_adam --num_epochs 100 --normalize_kl --normalize_nll --tune_on_nll --val_teacher_forcing --teacher_forcing_steps -1'
    mkdir -p $WORKING_DIR
    bsub -J "bball" -n $NUM_CPU_CORES -W $RUN_TIME -R "rusage[mem=$MEM_PER_CPU_CORE,ngpus_excl_p=$NUM_GPU_CORES]" -R "select[gpu_mtotal0>=10240]" <<ENDBSUB
    python -u dnri/experiments/bball_experiment.py --gpu --mode train --data_path $DATA_PATH --working_dir $WORKING_DIR $MODEL_ARGS $TRAINING_ARGS |& tee "${WORKING_DIR}results.txt"
    python -u dnri/experiments/bball_experiment.py --gpu --mode eval --load_best_model --data_path $DATA_PATH --working_dir $WORKING_DIR $MODEL_ARGS $TRAINING_ARGS |& tee "${WORKING_DIR}eval_results.txt"
ENDBSUB
    ###### DNRI – 4
    WORKING_DIR="${BASE_RESULTS_DIR}dnri_4_edge/seed_${SEED}/"
    ENCODER_ARGS="--encoder_hidden 256 --encoder_mlp_num_layers 3 --encoder_mlp_hidden 128 --encoder_rnn_hidden 64"
    DECODER_ARGS="--decoder_hidden 256"
    HIDDEN_ARGS="--rnn_hidden 64"
    PRIOR_ARGS="--use_learned_prior --prior_num_layers 3 --prior_hidden_size 128"
    MODEL_ARGS="--model_type dnri --graph_type dynamic --num_edge_types 4 $ENCODER_ARGS $DECODER_ARGS $HIDDEN_ARGS $PRIOR_ARGS --seed ${SEED}"
    TRAINING_ARGS='--batch_size 56 --lr 5e-4 --use_adam --num_epochs 100 --normalize_kl --normalize_nll --tune_on_nll --val_teacher_forcing --teacher_forcing_steps -1'
    mkdir -p $WORKING_DIR
    bsub -J "bball" -n $NUM_CPU_CORES -W $RUN_TIME -R "rusage[mem=$MEM_PER_CPU_CORE,ngpus_excl_p=$NUM_GPU_CORES]" -R "select[gpu_mtotal0>=10240]" <<ENDBSUB
    python -u dnri/experiments/bball_experiment.py --gpu --mode train --data_path $DATA_PATH --working_dir $WORKING_DIR $MODEL_ARGS $TRAINING_ARGS |& tee "${WORKING_DIR}results.txt"
    python -u dnri/experiments/bball_experiment.py --gpu --mode eval --load_best_model --data_path $DATA_PATH --working_dir $WORKING_DIR $MODEL_ARGS $TRAINING_ARGS |& tee "${WORKING_DIR}eval_results.txt"
ENDBSUB
    ###### DNRI – 6, skip_first
    WORKING_DIR="${BASE_RESULTS_DIR}dnri_6_edge_skip_first/seed_${SEED}/"
    ENCODER_ARGS="--encoder_hidden 256 --encoder_mlp_num_layers 3 --encoder_mlp_hidden 128 --encoder_rnn_hidden 64"
    DECODER_ARGS="--decoder_hidden 256"
    HIDDEN_ARGS="--rnn_hidden 64"
    PRIOR_ARGS="--use_learned_prior --prior_num_layers 3 --prior_hidden_size 128"
    MODEL_ARGS="--model_type dnri --graph_type dynamic --skip_first --num_edge_types 6 $ENCODER_ARGS $DECODER_ARGS $HIDDEN_ARGS $PRIOR_ARGS --seed ${SEED}"
    TRAINING_ARGS='--batch_size 56 --lr 5e-4 --use_adam --num_epochs 100 --normalize_kl --normalize_nll --tune_on_nll --val_teacher_forcing --teacher_forcing_steps -1'
    mkdir -p $WORKING_DIR
    bsub -J "bball" -n $NUM_CPU_CORES -W $RUN_TIME -R "rusage[mem=$MEM_PER_CPU_CORE,ngpus_excl_p=$NUM_GPU_CORES]" -R "select[gpu_mtotal0>=10240]" <<ENDBSUB
    python -u dnri/experiments/bball_experiment.py --gpu --mode train --data_path $DATA_PATH --working_dir $WORKING_DIR $MODEL_ARGS $TRAINING_ARGS |& tee "${WORKING_DIR}results.txt"
    python -u dnri/experiments/bball_experiment.py --gpu --mode eval --load_best_model --data_path $DATA_PATH --working_dir $WORKING_DIR $MODEL_ARGS $TRAINING_ARGS |& tee "${WORKING_DIR}eval_results.txt"
ENDBSUB
    ###### DNRI – 6
    WORKING_DIR="${BASE_RESULTS_DIR}dnri_6_edge/seed_${SEED}/"
    ENCODER_ARGS="--encoder_hidden 256 --encoder_mlp_num_layers 3 --encoder_mlp_hidden 128 --encoder_rnn_hidden 64"
    DECODER_ARGS="--decoder_hidden 256"
    HIDDEN_ARGS="--rnn_hidden 64"
    PRIOR_ARGS="--use_learned_prior --prior_num_layers 3 --prior_hidden_size 128"
    MODEL_ARGS="--model_type dnri --graph_type dynamic --num_edge_types 6 $ENCODER_ARGS $DECODER_ARGS $HIDDEN_ARGS $PRIOR_ARGS --seed ${SEED}"
    TRAINING_ARGS='--batch_size 56 --lr 5e-4 --use_adam --num_epochs 100 --normalize_kl --normalize_nll --tune_on_nll --val_teacher_forcing --teacher_forcing_steps -1'
    mkdir -p $WORKING_DIR
    bsub -J "bball" -n $NUM_CPU_CORES -W $RUN_TIME -R "rusage[mem=$MEM_PER_CPU_CORE,ngpus_excl_p=$NUM_GPU_CORES]" -R "select[gpu_mtotal0>=10240]" <<ENDBSUB
    python -u dnri/experiments/bball_experiment.py --gpu --mode train --data_path $DATA_PATH --working_dir $WORKING_DIR $MODEL_ARGS $TRAINING_ARGS |& tee "${WORKING_DIR}results.txt"
    python -u dnri/experiments/bball_experiment.py --gpu --mode eval --load_best_model --data_path $DATA_PATH --working_dir $WORKING_DIR $MODEL_ARGS $TRAINING_ARGS |& tee "${WORKING_DIR}eval_results.txt"
ENDBSUB
    ###### DNRI – 8, skip_first
    WORKING_DIR="${BASE_RESULTS_DIR}dnri_8_edge_skip_first/seed_${SEED}/"
    ENCODER_ARGS="--encoder_hidden 256 --encoder_mlp_num_layers 3 --encoder_mlp_hidden 128 --encoder_rnn_hidden 64"
    DECODER_ARGS="--decoder_hidden 256"
    HIDDEN_ARGS="--rnn_hidden 64"
    PRIOR_ARGS="--use_learned_prior --prior_num_layers 3 --prior_hidden_size 128"
    MODEL_ARGS="--model_type dnri --graph_type dynamic --skip_first --num_edge_types 8 $ENCODER_ARGS $DECODER_ARGS $HIDDEN_ARGS $PRIOR_ARGS --seed ${SEED}"
    TRAINING_ARGS='--batch_size 56 --lr 5e-4 --use_adam --num_epochs 100 --normalize_kl --normalize_nll --tune_on_nll --val_teacher_forcing --teacher_forcing_steps -1'
    mkdir -p $WORKING_DIR
    bsub -J "bball" -n $NUM_CPU_CORES -W $RUN_TIME -R "rusage[mem=$MEM_PER_CPU_CORE,ngpus_excl_p=$NUM_GPU_CORES]" -R "select[gpu_mtotal0>=10240]" <<ENDBSUB
    python -u dnri/experiments/bball_experiment.py --gpu --mode train --data_path $DATA_PATH --working_dir $WORKING_DIR $MODEL_ARGS $TRAINING_ARGS |& tee "${WORKING_DIR}results.txt"
    python -u dnri/experiments/bball_experiment.py --gpu --mode eval --load_best_model --data_path $DATA_PATH --working_dir $WORKING_DIR $MODEL_ARGS $TRAINING_ARGS |& tee "${WORKING_DIR}eval_results.txt"
ENDBSUB
    ###### DNRI – 8
    WORKING_DIR="${BASE_RESULTS_DIR}dnri_8_edge/seed_${SEED}/"
    ENCODER_ARGS="--encoder_hidden 256 --encoder_mlp_num_layers 3 --encoder_mlp_hidden 128 --encoder_rnn_hidden 64"
    DECODER_ARGS="--decoder_hidden 256"
    HIDDEN_ARGS="--rnn_hidden 64"
    PRIOR_ARGS="--use_learned_prior --prior_num_layers 3 --prior_hidden_size 128"
    MODEL_ARGS="--model_type dnri --graph_type dynamic --num_edge_types 8 $ENCODER_ARGS $DECODER_ARGS $HIDDEN_ARGS $PRIOR_ARGS --seed ${SEED}"
    TRAINING_ARGS='--batch_size 56 --lr 5e-4 --use_adam --num_epochs 100 --normalize_kl --normalize_nll --tune_on_nll --val_teacher_forcing --teacher_forcing_steps -1'
    mkdir -p $WORKING_DIR
    bsub -J "bball" -n $NUM_CPU_CORES -W $RUN_TIME -R "rusage[mem=$MEM_PER_CPU_CORE,ngpus_excl_p=$NUM_GPU_CORES]" -R "select[gpu_mtotal0>=10240]" <<ENDBSUB
    python -u dnri/experiments/bball_experiment.py --gpu --mode train --data_path $DATA_PATH --working_dir $WORKING_DIR $MODEL_ARGS $TRAINING_ARGS |& tee "${WORKING_DIR}results.txt"
    python -u dnri/experiments/bball_experiment.py --gpu --mode eval --load_best_model --data_path $DATA_PATH --working_dir $WORKING_DIR $MODEL_ARGS $TRAINING_ARGS |& tee "${WORKING_DIR}eval_results.txt"
ENDBSUB

    ########################    NRI
    ###### NRI – 4, skip_first
    WORKING_DIR="${BASE_RESULTS_DIR}nri_4_edge_skip_first/seed_${SEED}/"
    ENCODER_ARGS='--num_edge_types 4 --encoder_hidden 256 --skip_first --encoder_mlp_hidden 256 --encoder_mlp_num_layers 3'
    DECODER_ARGS=''
    MODEL_ARGS="--model_type nri --graph_type static ${ENCODER_ARGS} ${DECODER_ARGS} --seed ${SEED}"
    TRAINING_ARGS='--batch_size 56 --lr 5e-4 --use_adam --num_epochs 100 --normalize_kl --normalize_nll --tune_on_nll --val_teacher_forcing'
    mkdir -p $WORKING_DIR
    DYNAMIC_MODEL_ARGS="--model_type nri --graph_type dynamic ${ENCODER_ARGS} ${DECODER_ARGS}"
    bsub -J "bball" -n $NUM_CPU_CORES -W $RUN_TIME -R "rusage[mem=$MEM_PER_CPU_CORE,ngpus_excl_p=$NUM_GPU_CORES]" -R "select[gpu_mtotal0>=10240]" <<ENDBSUB
    python -u dnri/experiments/bball_experiment.py --gpu --mode train --data_path $DATA_PATH --working_dir $WORKING_DIR $MODEL_ARGS $TRAINING_ARGS |& tee "${WORKING_DIR}results.txt"
    python -u dnri/experiments/bball_experiment.py --gpu --mode eval --load_best_model --data_path $DATA_PATH --working_dir $WORKING_DIR $MODEL_ARGS $TRAINING_ARGS |& tee "${WORKING_DIR}eval_results.txt"
    python -u dnri/experiments/bball_experiment.py --gpu --mode eval --load_best_model --data_path $DATA_PATH --working_dir $WORKING_DIR $DYNAMIC_MODEL_ARGS $TRAINING_ARGS --error_out_name prediction_errors_dynamic.npy |& tee "${WORKING_DIR}eval_results_dynamic.txt"
ENDBSUB
    ###### NRI – 4
    WORKING_DIR="${BASE_RESULTS_DIR}nri_4_edge/seed_${SEED}/"
    ENCODER_ARGS='--num_edge_types 4 --encoder_hidden 256 --encoder_mlp_hidden 256 --encoder_mlp_num_layers 3'
    DECODER_ARGS=''
    MODEL_ARGS="--model_type nri --graph_type static ${ENCODER_ARGS} ${DECODER_ARGS} --seed ${SEED}"
    TRAINING_ARGS='--batch_size 56 --lr 5e-4 --use_adam --num_epochs 100 --normalize_kl --normalize_nll --tune_on_nll --val_teacher_forcing'
    mkdir -p $WORKING_DIR
    DYNAMIC_MODEL_ARGS="--model_type nri --graph_type dynamic ${ENCODER_ARGS} ${DECODER_ARGS}"
    bsub -J "bball" -n $NUM_CPU_CORES -W $RUN_TIME -R "rusage[mem=$MEM_PER_CPU_CORE,ngpus_excl_p=$NUM_GPU_CORES]" -R "select[gpu_mtotal0>=10240]" <<ENDBSUB
    python -u dnri/experiments/bball_experiment.py --gpu --mode train --data_path $DATA_PATH --working_dir $WORKING_DIR $MODEL_ARGS $TRAINING_ARGS |& tee "${WORKING_DIR}results.txt"
    python -u dnri/experiments/bball_experiment.py --gpu --mode eval --load_best_model --data_path $DATA_PATH --working_dir $WORKING_DIR $MODEL_ARGS $TRAINING_ARGS |& tee "${WORKING_DIR}eval_results.txt"
    python -u dnri/experiments/bball_experiment.py --gpu --mode eval --load_best_model --data_path $DATA_PATH --working_dir $WORKING_DIR $DYNAMIC_MODEL_ARGS $TRAINING_ARGS --error_out_name prediction_errors_dynamic.npy |& tee "${WORKING_DIR}eval_results_dynamic.txt"
ENDBSUB
    ###### NRI – 6, skip_first
    WORKING_DIR="${BASE_RESULTS_DIR}nri_6_edge_skip_first/seed_${SEED}/"
    ENCODER_ARGS='--num_edge_types 6 --encoder_hidden 256 --skip_first --encoder_mlp_hidden 256 --encoder_mlp_num_layers 3'
    DECODER_ARGS=''
    MODEL_ARGS="--model_type nri --graph_type static ${ENCODER_ARGS} ${DECODER_ARGS} --seed ${SEED}"
    TRAINING_ARGS='--batch_size 56 --lr 5e-4 --use_adam --num_epochs 100 --normalize_kl --normalize_nll --tune_on_nll --val_teacher_forcing'
    mkdir -p $WORKING_DIR
    DYNAMIC_MODEL_ARGS="--model_type nri --graph_type dynamic ${ENCODER_ARGS} ${DECODER_ARGS}"
    bsub -J "bball" -n $NUM_CPU_CORES -W $RUN_TIME -R "rusage[mem=$MEM_PER_CPU_CORE,ngpus_excl_p=$NUM_GPU_CORES]" -R "select[gpu_mtotal0>=10240]" <<ENDBSUB
    python -u dnri/experiments/bball_experiment.py --gpu --mode train --data_path $DATA_PATH --working_dir $WORKING_DIR $MODEL_ARGS $TRAINING_ARGS |& tee "${WORKING_DIR}results.txt"
    python -u dnri/experiments/bball_experiment.py --gpu --mode eval --load_best_model --data_path $DATA_PATH --working_dir $WORKING_DIR $MODEL_ARGS $TRAINING_ARGS |& tee "${WORKING_DIR}eval_results.txt"
    python -u dnri/experiments/bball_experiment.py --gpu --mode eval --load_best_model --data_path $DATA_PATH --working_dir $WORKING_DIR $DYNAMIC_MODEL_ARGS $TRAINING_ARGS --error_out_name prediction_errors_dynamic.npy |& tee "${WORKING_DIR}eval_results_dynamic.txt"
ENDBSUB
    ###### NRI – 6
    WORKING_DIR="${BASE_RESULTS_DIR}nri_6_edge/seed_${SEED}/"
    ENCODER_ARGS='--num_edge_types 6 --encoder_hidden 256 --encoder_mlp_hidden 256 --encoder_mlp_num_layers 3'
    DECODER_ARGS=''
    MODEL_ARGS="--model_type nri --graph_type static ${ENCODER_ARGS} ${DECODER_ARGS} --seed ${SEED}"
    TRAINING_ARGS='--batch_size 56 --lr 5e-4 --use_adam --num_epochs 100 --normalize_kl --normalize_nll --tune_on_nll --val_teacher_forcing'
    mkdir -p $WORKING_DIR
    DYNAMIC_MODEL_ARGS="--model_type nri --graph_type dynamic ${ENCODER_ARGS} ${DECODER_ARGS}"
    bsub -J "bball" -n $NUM_CPU_CORES -W $RUN_TIME -R "rusage[mem=$MEM_PER_CPU_CORE,ngpus_excl_p=$NUM_GPU_CORES]" -R "select[gpu_mtotal0>=10240]" <<ENDBSUB
    python -u dnri/experiments/bball_experiment.py --gpu --mode train --data_path $DATA_PATH --working_dir $WORKING_DIR $MODEL_ARGS $TRAINING_ARGS |& tee "${WORKING_DIR}results.txt"
    python -u dnri/experiments/bball_experiment.py --gpu --mode eval --load_best_model --data_path $DATA_PATH --working_dir $WORKING_DIR $MODEL_ARGS $TRAINING_ARGS |& tee "${WORKING_DIR}eval_results.txt"
    python -u dnri/experiments/bball_experiment.py --gpu --mode eval --load_best_model --data_path $DATA_PATH --working_dir $WORKING_DIR $DYNAMIC_MODEL_ARGS $TRAINING_ARGS --error_out_name prediction_errors_dynamic.npy |& tee "${WORKING_DIR}eval_results_dynamic.txt"
ENDBSUB
    ###### NRI – 8, skip_first
    WORKING_DIR="${BASE_RESULTS_DIR}nri_8_edge_skip_first/seed_${SEED}/"
    ENCODER_ARGS='--num_edge_types 8 --encoder_hidden 256 --skip_first --encoder_mlp_hidden 256 --encoder_mlp_num_layers 3'
    DECODER_ARGS=''
    MODEL_ARGS="--model_type nri --graph_type static ${ENCODER_ARGS} ${DECODER_ARGS} --seed ${SEED}"
    TRAINING_ARGS='--batch_size 56 --lr 5e-4 --use_adam --num_epochs 100 --normalize_kl --normalize_nll --tune_on_nll --val_teacher_forcing'
    mkdir -p $WORKING_DIR
    DYNAMIC_MODEL_ARGS="--model_type nri --graph_type dynamic ${ENCODER_ARGS} ${DECODER_ARGS}"
    bsub -J "bball" -n $NUM_CPU_CORES -W $RUN_TIME -R "rusage[mem=$MEM_PER_CPU_CORE,ngpus_excl_p=$NUM_GPU_CORES]" -R "select[gpu_mtotal0>=10240]" <<ENDBSUB
    python -u dnri/experiments/bball_experiment.py --gpu --mode train --data_path $DATA_PATH --working_dir $WORKING_DIR $MODEL_ARGS $TRAINING_ARGS |& tee "${WORKING_DIR}results.txt"
    python -u dnri/experiments/bball_experiment.py --gpu --mode eval --load_best_model --data_path $DATA_PATH --working_dir $WORKING_DIR $MODEL_ARGS $TRAINING_ARGS |& tee "${WORKING_DIR}eval_results.txt"
    python -u dnri/experiments/bball_experiment.py --gpu --mode eval --load_best_model --data_path $DATA_PATH --working_dir $WORKING_DIR $DYNAMIC_MODEL_ARGS $TRAINING_ARGS --error_out_name prediction_errors_dynamic.npy |& tee "${WORKING_DIR}eval_results_dynamic.txt"
ENDBSUB
    ###### NRI – 8
    WORKING_DIR="${BASE_RESULTS_DIR}nri_8_edge/seed_${SEED}/"
    ENCODER_ARGS='--num_edge_types 8 --encoder_hidden 256 --encoder_mlp_hidden 256 --encoder_mlp_num_layers 3'
    DECODER_ARGS=''
    MODEL_ARGS="--model_type nri --graph_type static ${ENCODER_ARGS} ${DECODER_ARGS} --seed ${SEED}"
    TRAINING_ARGS='--batch_size 56 --lr 5e-4 --use_adam --num_epochs 100 --normalize_kl --normalize_nll --tune_on_nll --val_teacher_forcing'
    mkdir -p $WORKING_DIR
    DYNAMIC_MODEL_ARGS="--model_type nri --graph_type dynamic ${ENCODER_ARGS} ${DECODER_ARGS}"
    bsub -J "bball" -n $NUM_CPU_CORES -W $RUN_TIME -R "rusage[mem=$MEM_PER_CPU_CORE,ngpus_excl_p=$NUM_GPU_CORES]" -R "select[gpu_mtotal0>=10240]" <<ENDBSUB
    python -u dnri/experiments/bball_experiment.py --gpu --mode train --data_path $DATA_PATH --working_dir $WORKING_DIR $MODEL_ARGS $TRAINING_ARGS |& tee "${WORKING_DIR}results.txt"
    python -u dnri/experiments/bball_experiment.py --gpu --mode eval --load_best_model --data_path $DATA_PATH --working_dir $WORKING_DIR $MODEL_ARGS $TRAINING_ARGS |& tee "${WORKING_DIR}eval_results.txt"
    python -u dnri/experiments/bball_experiment.py --gpu --mode eval --load_best_model --data_path $DATA_PATH --working_dir $WORKING_DIR $DYNAMIC_MODEL_ARGS $TRAINING_ARGS --error_out_name prediction_errors_dynamic.npy |& tee "${WORKING_DIR}eval_results_dynamic.txt"
ENDBSUB
done
