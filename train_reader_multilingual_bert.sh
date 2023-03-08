# Download the dataset
data=$1
working_directory=$(pwd)

if [ -z $data ]; then
	echo "Requried to provide data to run training with==> {nq, trivia, squad1, all}"
	exit
fi


declare -a natural_questions_gold=("data.gold_passages_info.nq_train" "data.gold_passages_info.nq_dev" "data.gold_passages_info.nq_test")
declare -a natural_questions_retriever=("data.retriever.nq-dev" "data.retriever.nq-train")
declare -a trivia_retriever=("data.retriever.trivia-dev" "data.retriever.trivia-train")
declare -a squad1_retriever=("data.retriever.squad1-dev" "data.retriever.squad1-train")

mkdir -p checkpoint_path/reader/$data/multilingual_bert
mkdir -p reader_data/gold
for resource in "${natural_questions_gold[@]}"; do
    python3 dpr/data/download_data.py \
        --resource $resource \
        --output_dir reader_data/gold
done

if [ $data = "nq" ]; then
    mkdir -p reader_data/nq_retriever_results
    for resource in "${natural_questions_retriever[@]}"; do
        python3 dpr/data/download_data.py \
            --resource $resource \
            --output_dir reader_data/nq_retriever_results
    done
elif [ $data = "trivia" ]; then
    mkdir -p reader_data/trivia_retriever_results
    for resource in "${trivia_retriever[@]}"; do
        python3 dpr/data/download_data.py \
            --resource $resource \
            --output_dir reader_data/trivia_retriever_results
    done
elif [ $data = "squad1" ]; then
    mkdir -p reader_data/squad1_retriever_results
    for resource in "${squad1_retriever[@]}"; do
        python3 dpr/data/download_data.py \
            --resource $resource \
            --output_dir reader_data/squad1_retriever_results
    done
elif [ $data = "all" ]; then
    mkdir -p reader_data/nq_retriever_results
    for resource in "${natural_questions_retriever[@]}"; do
        python3 dpr/data/download_data.py \
            --resource $resource \
            --output_dir reader_data/nq_retriever_results
    done
    mkdir -p reader_data/trivia_retriever_results
    for resource in "${trivia_retriever[@]}"; do
        python3 dpr/data/download_data.py \
            --resource $resource \
            --output_dir reader_data/trivia_retriever_results
    done
    mkdir -p reader_data/squad1_retriever_results
    for resource in "${squad1_retriever[@]}"; do
        python3 dpr/data/download_data.py \
            --resource $resource \
            --output_dir reader_data/squad1_retriever_results
    done
fi

# # Train the model
if [ $data = "all" ]; then
    CUDA_VISIBLE_DEVICES=4,5,6,7 python train_extractive_reader.py \
        encoder.encoder_model_type=hf_bert \
        encoder.pretrained_model_cfg=bert-base-multilingual-uncased \
        encoder.sequence_length=350 \
        train.batch_size=16 \
        train_files=${working_directory}/reader_data/nq_retriever_results/downloads/data/retriever/nq-train.json,${working_directory}/reader_data/trivia_retriever_results/downloads/data/retriever/trivia-train.json,${working_directory}/reader_data/trivia_retriever_results/downloads/data/retriever/squad1-train.json \
        dev_files=${working_directory}/reader_data/nq_retriever_results/downloads/data/retriever/nq-dev.json,${working_directory}/reader_data/trivia_retriever_results/downloads/data/retriever/trivia-dev.json,${working_directory}/reader_data/trivia_retriever_results/downloads/data/retriever/squad1-dev.json \
        gold_passages_src=${working_directory}/reader_data/gold/downloads/data/gold_passages_info/nq_train.json \
        gold_passages_src_dev=${working_directory}/reader_data/gold/downloads/data/gold_passages_info/nq_dev.json \
        output_dir=checkpoint_path/reader/$data/multilingual_bert
else
    CUDA_VISIBLE_DEVICES=4,5,6,7 python train_extractive_reader.py \
        encoder.encoder_model_type=hf_bert \
        encoder.pretrained_model_cfg=bert-base-multilingual-uncased \
        encoder.sequence_length=350 \
        train.batch_size=16 \
        train_files=${working_directory}/reader_data/${data}_retriever_results/downloads/data/retriever/$data-train.json \
        dev_files=${working_directory}/reader_data/${data}_retriever_results/downloads/data/retriever/$data-dev.json \
        gold_passages_src=${working_directory}/reader_data/gold/downloads/data/gold_passages_info/nq_train.json \
        gold_passages_src_dev=${working_directory}/reader_data/gold/downloads/data/gold_passages_info/nq_dev.json \
        output_dir=checkpoint_path/reader/$data/multilingual_bert
fi
