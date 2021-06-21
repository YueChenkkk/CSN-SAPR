ROOT_DIR=""
BERT_PRETRAINED_DIR=""
CHECKPOINT_DIR=""
DATA_PREFIX="./data"

source ${ROOT_DIR}/.bashrc

CUDA_VISIBLE_DEVICES=2 python train.py \
--model_name CSN \
--pooling_type max_pooling \
--dropout 0.5 \
--optimizer adam \
--margin 1.0 \
--lr 2e-5 \
--num_epochs 50 \
--batch_size 16 \
--patience 10 \
--bert_pretrained_dir ${BERT_PRETRAINED_DIR} \
--train_file \
${DATA_PREFIX}/train/train_unsplit.txt \
--dev_file \
${DATA_PREFIX}/dev/dev_unsplit.txt \
--test_file \
${DATA_PREFIX}/test/test_unsplit.txt \
--name_list_path \
${DATA_PREFIX}/name_list.txt \
--length_limit 510 \
--checkpoint_dir ${CHECKPOINT_DIR}