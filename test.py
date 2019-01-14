# -*- coding=utf-8 -*-

import pickle
import pdb
import json

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

with open("token2idx.json", "r", encoding="utf-8") as f:
  token2idx = json.load(f)
with open("tag2idx.json", "r", encoding="utf-8") as f:
  tag2idx = json.load(f)

with open("initial_np.pkl", "rb") as f:
  initial_np = pickle.load(f)
with open("transit_np.pkl", "rb") as f:
  transit_np = pickle.load(f)
with open("emit_np.pkl", "rb") as f:
  emit_np = pickle.load(f)

def viterbi_decode(seq):
  best_ptr = np.zeros((len(seq), transit_np.shape[0]), dtype=int)
  alpha = initial_np * emit_np[:, seq[0]]
  for idx, tok_id in enumerate(seq[1:]):
    alpha_1 = np.expand_dims(alpha, 1) * transit_np
    best_ptr[idx + 1] = np.argmax(alpha_1, 0)
    alpha = np.max(alpha_1, 0)
    alpha = alpha * emit_np[:, tok_id]
  best_tag = np.argmax(alpha, 0)
  best_tag_record = [best_tag]
  for i in range(len(seq) - 1, 0, -1):
    best_tag = best_ptr[i, best_tag]
    best_tag_record.append(best_tag)
  return best_tag_record[::-1]

idx2tag = {v:k for k, v in tag2idx.items()}
pred = open("pred.txt", "w", encoding="utf-8")
with open("data/test.txt", "r", encoding="utf-8") as test:
  hyps = []
  labels = []
  for line in test.readlines():
    tokens = []
    tags = []
    for pair in line.strip().split(" "):
      slash = pair.rfind("/")
      token, tag = pair[:slash], pair[slash + 1:]
      if token == "" or tag == "":
        continue
      tokens.append(token2idx[token])
      tags.append(tag2idx[tag])
    hyp = viterbi_decode(tokens)
    assert len(hyp) == len(tags), "tags and hyp should be equal length."
    pred.write(" ".join(map(lambda x:idx2tag[x], hyp)) + "\n")
    hyps.extend(hyp)
    labels.extend(tags)
  print("micro-f1 score: {}".format(f1_score(labels, hyps, average="micro")))
  print("precision score: {}".format(precision_score(labels, hyps, average="micro")))
  print("recall score: {}".format(recall_score(labels, hyps, average="micro")))