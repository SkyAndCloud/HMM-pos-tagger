# -*- coding=utf-8 -*-

import os
import json
import pickle

import numpy as np

token2idx = {}
tag2idx = {}

with open(os.path.join("data", "raw_data.txt"), "r", encoding="utf-8") as data:
  for line in data.readlines():
    for pair in line.strip().split(" "):
      slash = pair.rfind("/")
      token, tag = pair[:slash], pair[slash + 1:]
      if token == "" or tag == "":
        continue
      if token not in token2idx:
        token2idx[token] = len(token2idx)
      if tag not in tag2idx:
        tag2idx[tag] = len(tag2idx)
with open("token2idx.json", "w", encoding="utf-8") as f:
  json.dump(token2idx, f, ensure_ascii=False)
with open("tag2idx.json", "w", encoding="utf-8") as f:
  json.dump(tag2idx, f, ensure_ascii=False)

initial_list = [0 for i in range(len(tag2idx))]
transit_list = [[0 for j in range(len(tag2idx))] for i in range(len(tag2idx))]
emit_list = [[0 for j in range(len(token2idx))] for i in range(len(tag2idx))]

with open(os.path.join("data", "train.txt"), "r", encoding="utf-8") as train:
  for line in train.readlines():
    prev_tag = -1
    for pair in line.strip().split(" "):
      slash = pair.rfind("/")
      token, tag = pair[:slash], pair[slash + 1:]
      if token == "" or tag == "":
        continue
      if prev_tag == -1:
        initial_list[tag2idx[tag]] += 1
      else:
        transit_list[tag2idx[prev_tag]][tag2idx[tag]] += 1
      emit_list[tag2idx[tag]][token2idx[token]] += 1
      prev_tag = tag

initial_np = np.array(initial_list)
transit_np = np.array(transit_list)
emit_np = np.array(emit_list)
initial_np = initial_np / initial_np.sum()
transit_np = transit_np / transit_np.sum(axis=1, keepdims=True)
transit_np[np.isnan(transit_np)] = 0
emit_np = emit_np / emit_np.sum(axis=1, keepdims=True)
emit_np[np.isnan(emit_np)] = 0

with open("initial_np.pkl", "wb") as f:
  pickle.dump(initial_np, f)
with open("transit_np.pkl", "wb") as f:
  pickle.dump(transit_np, f)
with open("emit_np.pkl", "wb") as f:
  pickle.dump(emit_np, f)