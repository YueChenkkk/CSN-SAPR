# CSN module definition

import torch.nn as nn
import torch.nn.functional as functional
import torch
from transformers import AutoModel


def get_nonlinear(nonlinear):
    """
    Activation function.
    """
    nonlinear_dict = {'relu':nn.ReLU(), 'tanh':nn.Tanh(), 'sigmoid':nn.Sigmoid(), 'softmax':nn.Softmax(dim=-1)}
    try:
        return nonlinear_dict[nonlinear]
    except:
        raise ValueError('not a valid nonlinear type!')


class SeqPooling(nn.Module):
    """
    Sequence pooling module.

    Can do max-pooling, mean-pooling and attentive-pooling on a list of sequences of different lengths.
    """
    def __init__(self, pooling_type, hidden_dim):
        super(SeqPooling, self).__init__()
        self.pooling_type = pooling_type
        self.hidden_dim = hidden_dim
        if pooling_type == 'attentive_pooling':
            self.query_vec = nn.parameter.Parameter(torch.randn(hidden_dim))

    def max_pool(self, seq):
        return seq.max(0)[0]

    def mean_pool(self, seq):
        return seq.mean(0)

    def attn_pool(self, seq):
        attn_score = torch.mm(seq, self.query_vec.view(-1, 1)).view(-1)
        attn_w = nn.Softmax(dim=0)(attn_score)
        weighted_sum = torch.mm(attn_w.view(1, -1), seq).view(-1)     
        return weighted_sum

    def forward(self, batch_seq):
        pooling_fn = {'max_pooling': self.max_pool,
                      'mean_pooling': self.mean_pool,
                      'attentive_pooling': self.attn_pool}
        pooled_seq = [pooling_fn[self.pooling_type](seq) for seq in batch_seq]
        return torch.stack(pooled_seq, dim=0)


class MLP_Scorer(nn.Module):
    """
    MLP scorer module.

    A perceptron with two layers.
    """
    def __init__(self, args, classifier_input_size):
        super(MLP_Scorer, self).__init__()
        self.scorer = nn.ModuleList()

        self.scorer.append(nn.Linear(classifier_input_size, args.classifier_intermediate_dim))
        self.scorer.append(nn.Linear(args.classifier_intermediate_dim, 1))
        self.nonlinear = get_nonlinear(args.nonlinear_type)

    def forward(self, x):
        for model in self.scorer:
            x = self.nonlinear(model(x))
        return x


class CSN(nn.Module):
    """
    Candidate Scoring Network.

    It's built on BERT with an MLP and other simple components.
    """
    def __init__(self, args):
        super(CSN, self).__init__()
        self.args = args
        self.bert_model = AutoModel.from_pretrained(args.bert_pretrained_dir)
        self.pooling = SeqPooling(args.pooling_type, self.bert_model.config.hidden_size)
        self.mlp_scorer = MLP_Scorer(args, self.bert_model.config.hidden_size * 3)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, features, sent_char_lens, mention_poses, quote_idxes, true_index, device):
        """
        params
            features: the candidate-specific segments (CSS) converted into the form of BERT input.  
            sent_char_lens: character-level lengths of sentences in CSSs.
                [[character-level length of sentence 1,...] in the CSS of candidate 1,...]
            mention_poses: the positions of the nearest candidate mentions.
                [(sentence-level index of nearest mention in CSS, 
                 character-level index of the leftmost character of nearest mention in CSS, 
                 character-level index of the rightmost character + 1) of candidate 1,...]
            quote_idxes: the sentence-level index of the quotes in CSSs.
                [index of quote in the CSS of candidate 1,...]
            true_index: the index of the true speaker.
            device: gpu/tpu/cpu device.
        """
        # encoding
        qs_hid = []
        ctx_hid = []
        cdd_hid = []
        for i, (cdd_sent_char_lens, cdd_mention_pos, cdd_quote_idx) in enumerate(zip(sent_char_lens, mention_poses, quote_idxes)):

            bert_output = self.bert_model(torch.tensor([features[i].input_ids], dtype=torch.long).to(device), token_type_ids=None, 
                attention_mask=torch.tensor([features[i].input_mask], dtype=torch.long).to(device))

            accum_char_len = [0]
            for sent_idx in range(len(cdd_sent_char_lens)):
                accum_char_len.append(accum_char_len[-1] + cdd_sent_char_lens[sent_idx])
            
            CSS_hid = bert_output['last_hidden_state'][0][1:sum(cdd_sent_char_lens) + 1]
            qs_hid.append(CSS_hid[accum_char_len[cdd_quote_idx]:accum_char_len[cdd_quote_idx + 1]])

            if len(cdd_sent_char_lens) == 1:
                ctx_hid.append(torch.zeros(1, CSS_hid.size(1)).to(device))
            elif cdd_mention_pos[0] == 0:
                ctx_hid.append(CSS_hid[:accum_char_len[-2]])
            else:
                ctx_hid.append(CSS_hid[accum_char_len[1]:])
            
            cdd_hid.append(CSS_hid[cdd_mention_pos[1]:cdd_mention_pos[2]])

        # pooling
        qs_rep = self.pooling(qs_hid)
        ctx_rep = self.pooling(ctx_hid)
        cdd_rep = self.pooling(cdd_hid)

        # concatenate
        feature_vector = torch.cat([qs_rep, ctx_rep, cdd_rep], dim=-1)

        # dropout
        feature_vector = self.dropout(feature_vector)
        
        # scoring
        scores = self.mlp_scorer(feature_vector).view(-1)
        scores_false = [scores[i] for i in range(scores.size(0)) if i != true_index]
        scores_true = [scores[true_index] for i in range(scores.size(0) - 1)]

        return scores, scores_false, scores_true

        