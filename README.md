# Introduction
  This is the source code of the paper *A Neural-network-based Approach to Identifying Speakers in Novels* presented on Interspeech 2021. It demonstrates the method to detect the speakers of the utterances in the novel given a fix range of contexts. Speaker-alternation-based revision is implemented in the evaluation script. A copy of the dataset on https://github.com/YueChenkkk/Chinese-Dataset-Speaker-Identification is included in ./data directory.

# Usage
  Training: bash run_train.sh  
  Evaluation: python evaluate.py
  
# Requirements
  python==3.6  
  jieba==0.42.1  
  numpy==1.19.1  
  torch==1.2.0  
  tensorflow==2.0.0  
  transformers==4.6.1  
  fastprogress==1.0.0
