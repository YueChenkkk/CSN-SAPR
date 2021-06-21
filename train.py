# Training script

import os
import sys
import pickle
import json
import time
import copy
from fastprogress import master_bar, progress_bar
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from transformers import AutoTokenizer

from utils.arguments import get_train_args
from utils.data_prep import build_data_loader
from utils.load_name_list import get_alias2id
from utils.bert_features import *
from utils.training_control import *
from model.model import CSN


# training log
LOG_FORMAT = '%(asctime)s %(name)s %(levelname)s %(pathname)s %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%m:%s %a'


def train():
    """
    Training script.

    return
        best_dev_acc: the best development accuracy.
        best_test_acc: the accuracy on test instances of the model that has the best performance on development instances.
    """
    args = get_train_args()
    timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())

    print("#######################OPTIONS########################")
    print(json.dumps(vars(args), indent=4))

    # checkpoint
    checkpoint_dir = os.path.join(args.checkpoint_dir, 
                                  os.path.join(args.model_name, timestamp))

    # logging
    writer = SummaryWriter(log_dir=os.path.join(checkpoint_dir, 'tensorboard'))

    logging_name = os.path.join(checkpoint_dir, 'training_log.log')
    logging.basicConfig(level=logging.INFO,
                        format=LOG_FORMAT,
                        datefmt=DATE_FORMAT,
                        filename=logging_name)

    # device
    device = torch.device('cuda:0')

    # data files
    train_file = args.train_file
    dev_file = args.dev_file
    test_file = args.test_file
    name_list_path = args.name_list_path

    alias2id = get_alias2id(name_list_path)

    # build training, development and test data loaders
    train_data = build_data_loader(train_file, alias2id, args, skip_only_one=True)
    print("The number of training instances: " + str(len(train_data)))
    dev_data = build_data_loader(dev_file, alias2id, args)
    print("The number of development instances: " + str(len(dev_data)))
    test_data = build_data_loader(test_file, alias2id, args)
    print("The number of test instances: " + str(len(test_data)))

    # example
    print('##############DEV EXAMPLE#################')
    dev_test_iter = iter(dev_data)
    _, CSSs, sent_char_lens, mention_poses, quote_idxes, one_hot_label, true_index, category = dev_test_iter.next()
    print('Candidate-specific segments:')
    print(CSSs)
    print('Nearest mention positions:')
    print(mention_poses)
    test_test_iter = iter(test_data)
    print('##############TEST EXAMPLE#################')
    _, CSSs, sent_char_lens, mention_poses, quote_idxes, one_hot_label, true_index, category = test_test_iter.next()
    print('Candidate-specific segments:')
    print(CSSs)
    print('Nearest mention positions:')
    print(mention_poses)

    # initialize model
    tokenizer = AutoTokenizer.from_pretrained(args.bert_pretrained_dir)
    model = CSN(args)
    model = model.to(device)

    # initialize optimizer
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        raise ValueError("Unknown optimizer type...")

    # loss criterion
    loss_fn = nn.MarginRankingLoss(margin=args.margin)

    # training loop
    print("############################Training Begins...################################")

    # logging best
    best_overall_dev_acc = 0
    best_explicit_dev_acc = 0
    best_implicit_dev_acc = 0
    best_latent_dev_acc = 0
    best_dev_loss = 0
    new_best = False

    # control parameters
    patience_counter = 0
    backward_counter = 0

    epoch_bar = master_bar(range(args.num_epochs))
    for epoch in epoch_bar:
        acc_numerator = 0
        acc_denominator = 0
        train_loss = 0

        model.train()
        optimizer.zero_grad()

        print('Epoch: %d' % (epoch + 1))
        for i, (_, CSSs, sent_char_lens, mention_poses, quote_idxes, one_hot_label, true_index, _) \
            in enumerate(progress_bar(train_data, total=len(train_data), parent=epoch_bar)):
            
            try:
                features = convert_examples_to_features(examples=CSSs, tokenizer=tokenizer)
                scores, scores_false, scores_true = model(features, sent_char_lens, mention_poses, quote_idxes, true_index, device)

                # backward propagation and weights update
                for x, y in zip(scores_false, scores_true):
                    # compute loss
                    loss = loss_fn(x.unsqueeze(0), y.unsqueeze(0), torch.tensor(-1.0).unsqueeze(0).to(device))
                    train_loss += loss.item()
                    
                    # backward propagation
                    loss /= args.batch_size
                    loss.backward(retain_graph=True)
                    backward_counter += 1

                    # update parameters
                    if backward_counter % args.batch_size == 0:
                        optimizer.step()
                        optimizer.zero_grad()

                # training accuracy
                acc_numerator += 1 if scores.max(0)[1].item() == true_index else 0
                acc_denominator += 1

            except RuntimeError:
                print('OOM occurs...')

        acc = acc_numerator / acc_denominator
        train_loss /= len(train_data)

        # logging
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', acc, epoch)
        logging.info('train_acc: %.4f' % (acc))
        print('train_acc: %.4f' % (acc))
        print('train_loss: %.4f' % (train_loss))

        # adjust learning rate after each epoch
        adjust_learning_rate(optimizer, args.lr_decay)

        # Evaluation
        model.eval()

        def eval(eval_data, subset_name):
            """
            Evaluate performance on a given subset.

            params
                eval_data: the set of instances to be evaluate on.
                subset_name: the name of the subset for logging.

            return
                acc_numerator_sub: the number of correct predictions.
                acc_denominator_sub: the total number of instances.
                sum_loss: the sum of evaluation loss on positive-negative pairs.
            """
            overall_eval_acc_numerator = 0
            overall_eval_acc_denominator = len(eval_data)
            explicit_eval_acc_numerator = 0
            explicit_eval_acc_denominator = 0
            implicit_eval_acc_numerator = 0
            implicit_eval_acc_denominator = 0
            latent_eval_acc_numerator = 0
            latent_eval_acc_denominator = 0

            eval_sum_loss = 0

            for _, CSSs, sent_char_lens, mention_poses, quote_idxes, _, true_index, category \
                in progress_bar(eval_data, total=len(eval_data), parent=epoch_bar):
                
                with torch.no_grad():
                    features = convert_examples_to_features(examples=CSSs, tokenizer=tokenizer)
                    scores, scores_false, scores_true = model(features, sent_char_lens, mention_poses, quote_idxes, true_index, device)
                    loss_list = [loss_fn(x.unsqueeze(0), y.unsqueeze(0), torch.tensor(-1.0).unsqueeze(0).to(device)) for x, y in zip(scores_false, scores_true)]
                
                eval_sum_loss += sum(x.item() for x in loss_list)

                # evaluate accuracy
                correct = 1 if scores.max(0)[1].item() == true_index else 0
                overall_eval_acc_numerator += correct
                if category == 'explicit':
                    explicit_eval_acc_numerator += correct
                    explicit_eval_acc_denominator += 1
                if category == 'implicit':
                    implicit_eval_acc_numerator += correct
                    implicit_eval_acc_denominator += 1
                if category == 'latent':
                    latent_eval_acc_numerator += correct
                    latent_eval_acc_denominator += 1

            overall_eval_acc = overall_eval_acc_numerator / overall_eval_acc_denominator
            explicit_eval_acc = explicit_eval_acc_numerator / explicit_eval_acc_denominator
            implicit_eval_acc = implicit_eval_acc_numerator / implicit_eval_acc_denominator
            latent_eval_acc = latent_eval_acc_numerator / latent_eval_acc_denominator
            eval_avg_loss = eval_sum_loss / overall_eval_acc_denominator

            # logging
            writer.add_scalar('Loss/' + subset_name, eval_avg_loss, epoch)
            writer.add_scalar('Accuracy/' + subset_name, overall_eval_acc, epoch)
            logging.info(subset_name + '_overall_acc: %.4f' % (overall_eval_acc))
            print(subset_name + '_overall_acc: %.4f' % (overall_eval_acc))
            print(subset_name + '_explicit_acc: %.4f' % (explicit_eval_acc))
            print(subset_name + '_implicit_acc: %.4f' % (implicit_eval_acc))
            print(subset_name + '_latent_acc: %.4f' % (latent_eval_acc))
            print(subset_name + '_overall_loss: %.4f' % (eval_avg_loss))

            return overall_eval_acc, explicit_eval_acc, implicit_eval_acc, latent_eval_acc, eval_avg_loss

        # development stage
        overall_dev_acc, explicit_dev_acc, implicit_dev_acc, latent_dev_acc, dev_avg_loss = eval(dev_data, 'dev')

        # save the model with best performance
        if overall_dev_acc > best_overall_dev_acc:
            best_overall_dev_acc = overall_dev_acc
            best_explicit_dev_acc = explicit_dev_acc
            best_implicit_dev_acc = implicit_dev_acc
            best_latent_dev_acc = latent_dev_acc
            best_dev_loss = dev_avg_loss
            
            patience_counter = 0
            new_best = True
        else:
            patience_counter += 1
            new_best = False

        # only save the model which outperforms the former best on development set
        if new_best:
            # test stage
            overall_test_acc, explicit_test_acc, implicit_test_acc, latent_test_acc, test_avg_loss = eval(test_data, 'test')
            try:
                save_checkpoint({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()},
                    {
                    'args': vars(args),
                    'training_loss': train_loss,
                    'best_overall_dev_acc': best_overall_dev_acc,
                    'best_explicit_dev_acc': best_explicit_dev_acc,
                    'best_implicit_dev_acc': best_implicit_dev_acc,
                    'best_latent_dev_acc': best_latent_dev_acc,
                    'best_overall_dev_loss': best_dev_loss,
                    'overall_test_acc': overall_test_acc,
                    'explicit_test_acc': explicit_test_acc,
                    'implicit_test_acc': implicit_test_acc,
                    'latent_test_acc': latent_test_acc,
                    'overall_test_loss': test_avg_loss
                    },
                    checkpoint_dir)
            except Exception as e:
                print(e)

        # early stopping
        if patience_counter > args.patience:
            print("Early stopping...")
            break

        print('------------------------------------------------------')

    return best_overall_dev_acc, overall_test_acc


if __name__ == '__main__':
    # run several times and calculate average accuracy and standard deviation
    dev = []
    test = []
    for i in range(3):    
        dev_acc, test_acc = train()
        dev.append(dev_acc)
        test.append(test_acc)

    dev = np.array(dev)
    test = np.array(test)

    dev_mean = np.mean(dev)
    dev_std = np.std(dev)
    test_mean = np.mean(test)
    test_std = np.std(test)

    print(str(dev_mean) + '(±' + str(dev_std) + ')')
    print(str(test_mean) + '(±' + str(test_std) + ')')
