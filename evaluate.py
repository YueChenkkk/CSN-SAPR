
import os
import pickle
import json
from fastprogress import master_bar, progress_bar
from argparse import ArgumentParser

import numpy as np
from sklearn.metrics import accuracy_score
import torch
from transformers import AutoTokenizer

from utils.arguments import get_train_args
from utils.data_prep import build_data_loader
from utils.bert_features import *
from utils.training_control import *
from utils.load_name_list import *
from model.model import CSN
from sapr.sap_rev import Prediction, sap_rev


def evaluate(checkpoint_dir, eval_file_path):
    """
    arg
        checkpoint_dir: the saved checkpoint of CSN model
        eval_file_path: the file 

    output
        Evalution result files under checkpoint_dir
    """
    parser = ArgumentParser()
    args = parser.parse_args()
    with open(os.path.join(checkpoint_dir, 'info.json'), 'r', encoding='utf-8') as fin:
        args.__dict__ = json.load(fin)['args']

    print("#######################OPTIONS########################")
    print(json.dumps(vars(args), indent=4))

    device = torch.device('cuda:0')

    alias2id = get_alias2id(name_list_path)
    id2alias = get_id2alias(name_list_path)

    # build data loader on test instances
    test_data = build_data_loader(eval_file_path, alias2id, args)

    # example
    print('##############EXAMPLE#################')
    test_test_iter = iter(test_data)
    raw_sents_in_list, CSSs, sent_char_lens, mention_poses, quote_idxes, one_hot_label, true_index, category = test_test_iter.next()
    print('Candidate-specific segments:')
    print(CSSs)
    print('Nearest mention positions:')
    print(mention_poses)

    # initialize model
    tokenizer = AutoTokenizer.from_pretrained(args.bert_pretrained_dir)
    model = CSN(args)
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'csn.ckpt'), map_location='cpu')['model'])
    model = model.to(device)

    # Evaluation
    model.eval()

    pred_list = []
    ground_truth = []
    candidate_ids_list = []
    candidate_aliases_list = []
    inst_categores = []
    for seg_sents, CSSs, sent_char_lens, mention_poses, quote_idxes, _, true_index, category \
        in progress_bar(test_data, total=len(test_data)):

        with torch.no_grad():
            features = convert_examples_to_features(examples=CSSs, tokenizer=tokenizer)
            scores, _, _ = model(features, sent_char_lens, mention_poses, quote_idxes, true_index, device)

        # continuous conversation correction
        candidate_aliases = [CSS[cdd_pos[1]:cdd_pos[2]] for cdd_pos, CSS in zip(mention_poses, CSSs)]
        candidate_ids = [alias2id[x] for x in candidate_aliases]
        
        pred_list.append(Prediction(seg_sents, scores.detach().cpu().numpy(), candidate_ids))
        ground_truth.append(candidate_ids[true_index])
        candidate_ids_list.append(candidate_ids)
        candidate_aliases_list.append(candidate_aliases)
        inst_categores.append(category)
    
    # SAPR
    sap_rev_dict = sap_rev(pred_list, alias2id, th=2)

    # calculate statistics
    pred_speakers = []
    revised_speakers = []
    n_rev = false2correct = correct2false = correct2correct = false2false = 0
    n_explicit = n_implicit = n_latent = 0
    csn_explicit_correct = csn_implicit_correct = csn_latent_correct = 0
    sapr_explicit_correct = sapr_implicit_correct = sapr_latent_correct = 0
    for idx, (pred, true_spk_id, ic) in enumerate(zip(pred_list, ground_truth, inst_categores)):
        if idx in sap_rev_dict:
            n_rev += 1
            rev_spk_id = sap_rev_dict[idx]
            if pred.pred_speaker_id != true_spk_id and rev_spk_id == true_spk_id:
                false2correct += 1
            if pred.pred_speaker_id == true_spk_id and rev_spk_id != true_spk_id:
                correct2false += 1
            if pred.pred_speaker_id == true_spk_id and rev_spk_id == true_spk_id:
                correct2correct += 1
            if pred.pred_speaker_id != true_spk_id and rev_spk_id != true_spk_id:
                false2false += 1
        else:
            rev_spk_id = pred.pred_speaker_id
        revised_speakers.append(rev_spk_id)
        pred_speakers.append(pred.pred_speaker_id)

        # accuracies for different categories
        if ic == 'explicit':
            n_explicit += 1
            csn_explicit_correct += 1 if true_spk_id == pred.pred_speaker_id else 0
            sapr_explicit_correct += 1 if true_spk_id == rev_spk_id else 0
        elif ic == 'implicit':
            n_implicit += 1
            csn_implicit_correct += 1 if true_spk_id == pred.pred_speaker_id else 0
            sapr_implicit_correct += 1 if true_spk_id == rev_spk_id else 0
        else:
            n_latent += 1
            csn_latent_correct += 1 if true_spk_id == pred.pred_speaker_id else 0
            sapr_latent_correct += 1 if true_spk_id == rev_spk_id else 0

    # write prediction results to file
    fout = open(os.path.join(checkpoint_dir, 'csn_sapr_on_test.txt'), 'w', encoding='utf-8')
    for idx, (pred, pred_spk, rev_spk, cdd_ids, cdd_aliases, gt, ic) in \
        enumerate(zip(pred_list, pred_speakers, revised_speakers, candidate_ids_list, candidate_aliases_list, ground_truth, inst_categores)):
        fout.write('Raw instance (%s):\n'.format(ic))
        for i, sent in enumerate(pred.seg_sents):
            postfix = '\t------QUOTE\n' if i == 10 else '\n'
            fout.write(''.join(sent) + postfix)

        pred_spk_alias = cdd_aliases[cdd_ids.index(pred_spk)]
        true_spk_alias = cdd_aliases[cdd_ids.index(gt)]
        if rev_spk in cdd_ids:
            rev_spk_alias = cdd_aliases[cdd_ids.index(rev_spk)]
        else:
            rev_spk_alias = id2alias[rev_spk]

        if idx in sap_rev_dict:
            fout.write('---Revised---\n')
        fout.write('Candidate roles: ' + str(cdd_aliases) + '\n')
        fout.write('True role name: ' + true_spk_alias + '\n')
        fout.write('Predicted role name: ' + pred_spk_alias + '\n')
        fout.write('Revised role name: ' + rev_spk_alias + '\n')
        fout.write('----------------------------------------------------------------------------------------\n')
    fout.close()

    # test accuracy
    csn_acc = accuracy_score(ground_truth, pred_speakers)
    sapr_acc = accuracy_score(ground_truth, revised_speakers)
    csn_explicit_acc = csn_explicit_correct / n_explicit
    csn_implicit_acc = csn_implicit_correct / n_implicit
    csn_latent_acc = csn_latent_correct / n_latent
    sapr_explicit_acc = sapr_explicit_correct / n_explicit
    sapr_implicit_acc = sapr_implicit_correct / n_implicit
    sapr_latent_acc = sapr_latent_correct / n_latent

    log_list = ['################ Evaluation Results ##################',
                'Total revision: %d' % (n_rev),
                'False to correct: %d' % (false2correct),
                'Correct to false: %d' % (correct2false),
                'Correct to correct: %d' % (correct2correct),
                'False to false: %d' % (false2false),
                'Overall accuracy (CSN): %.4f' % (csn_acc),
                'Explicit accuracy (CSN): %.4f' % (csn_explicit_acc),
                'Implicit accuracy (CSN): %.4f' % (csn_implicit_acc),
                'Latent accuracy (CSN): %.4f' % (csn_latent_acc),
                'Overall accuracy (CSN + SAPR): %.4f' % (sapr_acc),
                'Explicit accuracy (CSN + SAPR): %.4f' % (sapr_explicit_acc),
                'Implicit accuracy (CSN + SAPR): %.4f' % (sapr_implicit_acc),
                'Latent accuracy (CSN + SAPR): %.4f' % (sapr_latent_acc)]
    
    for log in log_list:
        print(log)

    with open(os.path.join(checkpoint_dir, 'test_stat.txt'), 'w', encoding='utf-8') as fout:
        for log in log_list:
            fout.write(log + '\n')


if __name__ == '__main__':
    CHECKPOINT_DIR = ''
    TEST_FILE_PATH = './data/test/test_unsplit.txt'
    evaluate(CHECKPOINT_DIR, TEST_FILE_PATH)
