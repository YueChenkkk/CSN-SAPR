# Speaker-alternation-pattern-based revision

import copy
import numpy as np


class Prediction:
	def __init__(self, seg_sents, scores, candidate_ids):
		self.seg_sents = seg_sents
		self.pred_speaker_id = candidate_ids[int(np.argmax(scores))]
		self.cdd_scores = {k: v for k, v in zip(candidate_ids, scores)}


def is_quote(sentence):
	return sentence.startswith('“') and sentence.endswith('”')


def retrieve_continuous_quotes(pred_list):
	"""

	"""
	ws = (len(pred_list[0].seg_sents) - 1) // 2

	connecting = False
	ctn_quotes_list = []
	for i, pred in enumerate(pred_list[:-1]):
		this_inst = [''.join(x) for x in pred.seg_sents]
		next_inst = [''.join(x) for x in pred_list[i + 1].seg_sents]

		# not in connecting mode, start a new connection
		if not connecting:
			# each element in inst_id_quote_id: (the index of instance in the dataset, the index of the center quote of the instance within this group of continuous quotes)
			inst_id_quote_id = [(i, 0)]
			connecting = True

		# compute the number of continuous quotes on the right
		n_ctn_quotes = 0
		for sent in this_inst[ws + 1:]:
			if is_quote(sent):
				n_ctn_quotes += 1
			else:
				break

		# no more adjacent quotes
		if n_ctn_quotes == 0:
			if len(inst_id_quote_id) > 1:
				ctn_quotes_list.append(inst_id_quote_id)
			connecting = False
			continue

		# compute the distance between this_inst and next_inst
		distance = ws + 1
		for dist in range(1, ws + 1):
			if this_inst[dist:] == next_inst[:-dist]:
				distance = dist
				break

		if dist <= n_ctn_quotes:
			inst_id_quote_id.append((i + 1, inst_id_quote_id[-1][1] + dist))
		else:
			if len(inst_id_quote_id) > 1:
				ctn_quotes_list.append(inst_id_quote_id)
			connecting = False

	if len(inst_id_quote_id) > 1:
		ctn_quotes_list.append(inst_id_quote_id)

	return ctn_quotes_list


def dominant_speakers(seg_ctx, alias2id, th):
	"""

	"""
	char_freq = {}
	for word in seg_ctx:
		if word not in alias2id:
			continue
		if alias2id[word] in char_freq:
			char_freq[alias2id[word]] += 1
		else:
			char_freq[alias2id[word]] = 1

	sorted_char_freq = sorted(char_freq.items(), key=lambda x: -x[1])
	if len(sorted_char_freq) < 2:
		return None
	
	c1 = sorted_char_freq[0][0]
	c2 = sorted_char_freq[1][0]

	if len(sorted_char_freq) == 2:
		return (c1, c2)
	else:
		fc2 = sorted_char_freq[1][1]
		fc3 = sorted_char_freq[2][1]
		if fc2 >= fc3 + th:
			return (c1, c2)
		else:
			return None


def pred_cfd(pred, speakers):
	c1, c2 = speakers
	cdd2score = pred.cdd_scores
	c1_score = -1 if c1 not in cdd2score else cdd2score[c1]
	c2_score = -1 if c2 not in cdd2score else cdd2score[c2]
	pred_spk = c1 if c1_score > c2_score else c2
	score_diff = c1_score - c2_score
	pred_cfd = score_diff if score_diff >= 0 else -score_diff
	return pred_spk, pred_cfd


def sap_figure_out_speaker(ctn_quotes, side, spk_idx):
	
	if side == 'left':
		# sap-revision start from the left
		prev_spk_idx = spk_idx
	else:
		prev_spk_idx = spk_idx if ctn_quotes[-1][1] % 2 == 0 else (1 - spk_idx)
		
	inst2spk_idx = {ctn_quotes[0][0]: prev_spk_idx}
	prev_quote_idx = 0
	for inst_idx, quote_idx_in_cvs in ctn_quotes[1:]:
		# sap-revision start from the right
		if (quote_idx_in_cvs - prev_quote_idx) % 2 == 0:
			inst2spk_idx[inst_idx] = prev_spk_idx
		else:
			inst2spk_idx[inst_idx] = 1 - prev_spk_idx
			prev_spk_idx = 1 - prev_spk_idx
		prev_quote_idx = quote_idx_in_cvs
	
	return inst2spk_idx


def sap_rev(pred_list, alias2id, th):
	"""
	Speaker-alternation-pattern-based revision (SAPR)
	
	params
		pred_list: a list of Prediction object.

	return
		rev_dict: {instance index: revised speaker id}
	"""
	ws = (len(pred_list[0].seg_sents) - 1) // 2

	ctn_quotes_list = retrieve_continuous_quotes(pred_list)

	name_list_path = '/home/ychen/183/codes_and_scripts/IdentifySpeaker/CSN_SAPR/data/name_list.txt'
	with open(name_list_path, 'r', encoding='utf-8') as fin:
		name_lines = fin.readlines()
	id2alias = []
	for i, line in enumerate(name_lines):
		id2alias.append(line.strip().split()[1])

	rev_dict = {}
	for ctn_quotes in ctn_quotes_list:
		# run sap-rev for each conversation

		left_inst_idx, _ = ctn_quotes[0]
		right_inst_idx, quote_idx = ctn_quotes[-1]

		left_pred = pred_list[left_inst_idx]
		right_pred = pred_list[right_inst_idx]

		seg_left_ctx_sents = left_pred.seg_sents[:ws]
		seg_right_ctx_sents = right_pred.seg_sents[-ws:]

		seg_ctx = [y for x in seg_left_ctx_sents + seg_right_ctx_sents for y in x]

		# two dominant speaker ids
		speakers = dominant_speakers(seg_ctx, alias2id, th)
		if not speakers:
			continue
		
		left_pred_spk, left_cfd = pred_cfd(left_pred, speakers)
		right_pred_spk, right_cfd = pred_cfd(right_pred, speakers)

		for seg_sent in left_pred.seg_sents:
			print(''.join(seg_sent))
		print('Left pred:', id2alias[left_pred_spk], 'Cfd:', left_cfd)
		print('---SEP---')
		for seg_sent in right_pred.seg_sents:
			print(''.join(seg_sent))
		print('Right pred:', id2alias[right_pred_spk], 'Cfd:', right_cfd)
		print([id2alias[x] for x in speakers])
		print('----------------------------------------------')

		side = 'left' if left_cfd > right_cfd else 'right'
		spk_id = left_pred_spk if left_cfd > right_cfd else right_pred_spk
		sap_rev_res = sap_figure_out_speaker(ctn_quotes, side, speakers.index(spk_id))
		
		rev_dict.update({k: speakers[v] for k, v in sap_rev_res.items()})

	return rev_dict
