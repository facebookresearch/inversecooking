# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
from build_vocab import Vocabulary
import random
import json
import lmdb


class Recipe1MDataset(data.Dataset):

    def __init__(self, data_dir, aux_data_dir, split, maxseqlen, maxnuminstrs, maxnumlabels, maxnumims,
                 transform=None, max_num_samples=-1, use_lmdb=False, suff=''):

        self.ingrs_vocab = pickle.load(open(os.path.join(aux_data_dir, suff + 'recipe1m_vocab_ingrs.pkl'), 'rb'))
        self.instrs_vocab = pickle.load(open(os.path.join(aux_data_dir, suff + 'recipe1m_vocab_toks.pkl'), 'rb'))
        self.dataset = pickle.load(open(os.path.join(aux_data_dir, suff + 'recipe1m_'+split+'.pkl'), 'rb'))

        self.label2word = self.get_ingrs_vocab()

        self.use_lmdb = use_lmdb
        if use_lmdb:
            self.image_file = lmdb.open(os.path.join(aux_data_dir, 'lmdb_' + split), max_readers=1, readonly=True,
                                        lock=False, readahead=False, meminit=False)

        self.ids = []
        self.split = split
        for i, entry in enumerate(self.dataset):
            if len(entry['images']) == 0:
                continue
            self.ids.append(i)

        self.root = os.path.join(data_dir, 'images', split)
        self.transform = transform
        self.max_num_labels = maxnumlabels
        self.maxseqlen = maxseqlen
        self.max_num_instrs = maxnuminstrs
        self.maxseqlen = maxseqlen*maxnuminstrs
        self.maxnumims = maxnumims
        if max_num_samples != -1:
            random.shuffle(self.ids)
            self.ids = self.ids[:max_num_samples]

    def get_instrs_vocab(self):
        return self.instrs_vocab

    def get_instrs_vocab_size(self):
        return len(self.instrs_vocab)

    def get_ingrs_vocab(self):
        return [min(w, key=len) if not isinstance(w, str) else w for w in
                self.ingrs_vocab.idx2word.values()]  # includes 'pad' ingredient

    def get_ingrs_vocab_size(self):
        return len(self.ingrs_vocab)

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""

        sample = self.dataset[self.ids[index]]
        img_id = sample['id']
        captions = sample['tokenized']
        paths = sample['images'][0:self.maxnumims]

        idx = index

        labels = self.dataset[self.ids[idx]]['ingredients']
        title = sample['title']

        tokens = []
        tokens.extend(title)
        # add fake token to separate title from recipe
        tokens.append('<eoi>')
        for c in captions:
            tokens.extend(c)
            tokens.append('<eoi>')

        ilabels_gt = np.ones(self.max_num_labels) * self.ingrs_vocab('<pad>')
        pos = 0

        true_ingr_idxs = []
        for i in range(len(labels)):
            true_ingr_idxs.append(self.ingrs_vocab(labels[i]))

        for i in range(self.max_num_labels):
            if i >= len(labels):
                label = '<pad>'
            else:
                label = labels[i]
            label_idx = self.ingrs_vocab(label)
            if label_idx not in ilabels_gt:
                ilabels_gt[pos] = label_idx
                pos += 1

        ilabels_gt[pos] = self.ingrs_vocab('<end>')
        ingrs_gt = torch.from_numpy(ilabels_gt).long()

        if len(paths) == 0:
            path = None
            image_input = torch.zeros((3, 224, 224))
        else:
            if self.split == 'train':
                img_idx = np.random.randint(0, len(paths))
            else:
                img_idx = 0
            path = paths[img_idx]
            if self.use_lmdb:
                try:
                    with self.image_file.begin(write=False) as txn:
                        image = txn.get(path.encode())
                        image = np.fromstring(image, dtype=np.uint8)
                        image = np.reshape(image, (256, 256, 3))
                    image = Image.fromarray(image.astype('uint8'), 'RGB')
                except:
                    print ("Image id not found in lmdb. Loading jpeg file...")
                    image = Image.open(os.path.join(self.root, path[0], path[1],
                                                    path[2], path[3], path)).convert('RGB')
            else:
                image = Image.open(os.path.join(self.root, path[0], path[1], path[2], path[3], path)).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)
            image_input = image

        # Convert caption (string) to word ids.
        caption = []

        caption = self.caption_to_idxs(tokens, caption)
        caption.append(self.instrs_vocab('<end>'))

        caption = caption[0:self.maxseqlen]
        target = torch.Tensor(caption)

        return image_input, target, ingrs_gt, img_id, path, self.instrs_vocab('<pad>')

    def __len__(self):
        return len(self.ids)

    def caption_to_idxs(self, tokens, caption):

        caption.append(self.instrs_vocab('<start>'))
        for token in tokens:
            caption.append(self.instrs_vocab(token))
        return caption


def collate_fn(data):

    # Sort a data list by caption length (descending order).
    # data.sort(key=lambda x: len(x[2]), reverse=True)
    image_input, captions, ingrs_gt, img_id, path, pad_value = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).

    image_input = torch.stack(image_input, 0)
    ingrs_gt = torch.stack(ingrs_gt, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.ones(len(captions), max(lengths)).long()*pad_value[0]

    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    return image_input, targets, ingrs_gt, img_id, path


def get_loader(data_dir, aux_data_dir, split, maxseqlen,
               maxnuminstrs, maxnumlabels, maxnumims, transform, batch_size,
               shuffle, num_workers, drop_last=False,
               max_num_samples=-1,
               use_lmdb=False,
               suff=''):

    dataset = Recipe1MDataset(data_dir=data_dir, aux_data_dir=aux_data_dir, split=split,
                              maxseqlen=maxseqlen, maxnumlabels=maxnumlabels, maxnuminstrs=maxnuminstrs,
                              maxnumims=maxnumims,
                              transform=transform,
                              max_num_samples=max_num_samples,
                              use_lmdb=use_lmdb,
                              suff=suff)

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                              drop_last=drop_last, collate_fn=collate_fn, pin_memory=True)
    return data_loader, dataset
