# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch
import numpy as np
from args import get_parser
import pickle
import os
from torchvision import transforms
from build_vocab import Vocabulary
from model import get_model
from tqdm import tqdm
from data_loader import get_loader
import json
import sys
from model import mask_from_eos
import random
from utils.metrics import softIoU, update_error_types, compute_metrics
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
map_loc = None if torch.cuda.is_available() else 'cpu'


def compute_score(sampled_ids):

    if 1 in sampled_ids:
        cut = np.where(sampled_ids == 1)[0][0]
    else:
        cut = -1
    sampled_ids = sampled_ids[0:cut]
    score = float(len(set(sampled_ids))) / float(len(sampled_ids))

    return score


def label2onehot(labels, pad_value):

    # input labels to one hot vector
    inp_ = torch.unsqueeze(labels, 2)
    one_hot = torch.FloatTensor(labels.size(0), labels.size(1), pad_value + 1).zero_().to(device)
    one_hot.scatter_(2, inp_, 1)
    one_hot, _ = one_hot.max(dim=1)
    # remove pad and eos position
    one_hot = one_hot[:, 1:-1]
    one_hot[:, 0] = 0

    return one_hot


def main(args):

    where_to_save = os.path.join(args.save_dir, args.project_name, args.model_name)
    checkpoints_dir = os.path.join(where_to_save, 'checkpoints')
    logs_dir = os.path.join(where_to_save, 'logs')

    if not args.log_term:
        print ("Eval logs will be saved to:", os.path.join(logs_dir, 'eval.log'))
        sys.stdout = open(os.path.join(logs_dir, 'eval.log'), 'w')
        sys.stderr = open(os.path.join(logs_dir, 'eval.err'), 'w')

    vars_to_replace = ['greedy', 'recipe_only', 'ingrs_only', 'temperature', 'batch_size', 'maxseqlen',
                       'get_perplexity', 'use_true_ingrs', 'eval_split', 'save_dir', 'aux_data_dir',
                       'recipe1m_dir', 'project_name', 'use_lmdb', 'beam']
    store_dict = {}
    for var in vars_to_replace:
        store_dict[var] = getattr(args, var)
    args = pickle.load(open(os.path.join(checkpoints_dir, 'args.pkl'), 'rb'))
    for var in vars_to_replace:
        setattr(args, var, store_dict[var])
    print (args)

    transforms_list = []
    transforms_list.append(transforms.Resize((args.crop_size)))
    transforms_list.append(transforms.CenterCrop(args.crop_size))
    transforms_list.append(transforms.ToTensor())
    transforms_list.append(transforms.Normalize((0.485, 0.456, 0.406),
                                                (0.229, 0.224, 0.225)))
    # Image preprocessing
    transform = transforms.Compose(transforms_list)

    # data loader
    data_dir = args.recipe1m_dir
    data_loader, dataset = get_loader(data_dir, args.aux_data_dir, args.eval_split,
                                      args.maxseqlen, args.maxnuminstrs, args.maxnumlabels,
                                      args.maxnumims, transform, args.batch_size,
                                      shuffle=False, num_workers=args.num_workers,
                                      drop_last=False, max_num_samples=-1,
                                      use_lmdb=args.use_lmdb, suff=args.suff)

    ingr_vocab_size = dataset.get_ingrs_vocab_size()
    instrs_vocab_size = dataset.get_instrs_vocab_size()

    args.numgens = 1

    # Build the model
    model = get_model(args, ingr_vocab_size, instrs_vocab_size)
    model_path = os.path.join(args.save_dir, args.project_name, args.model_name, 'checkpoints', 'modelbest.ckpt')

    # overwrite flags for inference
    model.recipe_only = args.recipe_only
    model.ingrs_only = args.ingrs_only

    # Load the trained model parameters
    model.load_state_dict(torch.load(model_path, map_location=map_loc))

    model.eval()
    model = model.to(device)
    results_dict = {'recipes': {}, 'ingrs': {}, 'ingr_iou': {}}
    captions = {}
    iou = []
    error_types = {'tp_i': 0, 'fp_i': 0, 'fn_i': 0, 'tn_i': 0, 'tp_all': 0, 'fp_all': 0, 'fn_all': 0}
    perplexity_list = []
    n_rep, th = 0, 0.3

    for i, (img_inputs, true_caps_batch, ingr_gt, imgid, impath) in tqdm(enumerate(data_loader)):

        ingr_gt = ingr_gt.to(device)
        true_caps_batch = true_caps_batch.to(device)

        true_caps_shift = true_caps_batch.clone()[:, 1:].contiguous()
        img_inputs = img_inputs.to(device)

        true_ingrs = ingr_gt if args.use_true_ingrs else None
        for gens in range(args.numgens):
            with torch.no_grad():

                if args.get_perplexity:

                    losses = model(img_inputs, true_caps_batch, ingr_gt, keep_cnn_gradients=False)
                    recipe_loss = losses['recipe_loss']
                    recipe_loss = recipe_loss.view(true_caps_shift.size())
                    non_pad_mask = true_caps_shift.ne(instrs_vocab_size - 1).float()
                    recipe_loss = torch.sum(recipe_loss*non_pad_mask, dim=-1) / torch.sum(non_pad_mask, dim=-1)
                    perplexity = torch.exp(recipe_loss)

                    perplexity = perplexity.detach().cpu().numpy().tolist()
                    perplexity_list.extend(perplexity)

                else:

                    outputs = model.sample(img_inputs, args.greedy, args.temperature, args.beam, true_ingrs)

                    if not args.recipe_only:
                        fake_ingrs = outputs['ingr_ids']
                        pred_one_hot = label2onehot(fake_ingrs, ingr_vocab_size - 1)
                        target_one_hot = label2onehot(ingr_gt, ingr_vocab_size - 1)
                        iou_item = torch.mean(softIoU(pred_one_hot, target_one_hot)).item()
                        iou.append(iou_item)

                        update_error_types(error_types, pred_one_hot, target_one_hot)

                        fake_ingrs = fake_ingrs.detach().cpu().numpy()

                        for ingr_idx, fake_ingr in enumerate(fake_ingrs):

                            iou_item = softIoU(pred_one_hot[ingr_idx].unsqueeze(0),
                                               target_one_hot[ingr_idx].unsqueeze(0)).item()
                            results_dict['ingrs'][imgid[ingr_idx]] = []
                            results_dict['ingrs'][imgid[ingr_idx]].append(fake_ingr)
                            results_dict['ingr_iou'][imgid[ingr_idx]] = iou_item

                    if not args.ingrs_only:
                        sampled_ids_batch = outputs['recipe_ids']
                        sampled_ids_batch = sampled_ids_batch.cpu().detach().numpy()

                        for j, sampled_ids in enumerate(sampled_ids_batch):
                            score = compute_score(sampled_ids)
                            if score < th:
                                n_rep += 1
                            if imgid[j] not in captions.keys():
                                results_dict['recipes'][imgid[j]] = []
                                results_dict['recipes'][imgid[j]].append(sampled_ids)
    if args.get_perplexity:
        print (len(perplexity_list))
        print (np.mean(perplexity_list))
    else:

        if not args.recipe_only:
            ret_metrics = {'accuracy': [], 'f1': [], 'jaccard': [], 'f1_ingredients': []}
            compute_metrics(ret_metrics, error_types, ['accuracy', 'f1', 'jaccard', 'f1_ingredients'],
                            eps=1e-10,
                            weights=None)

            for k, v in ret_metrics.items():
                print (k, np.mean(v))

        if args.greedy:
            suff = 'greedy'
        else:
            if args.beam != -1:
                suff = 'beam_'+str(args.beam)
            else:
                suff = 'temp_' + str(args.temperature)

        results_file = os.path.join(args.save_dir, args.project_name, args.model_name, 'checkpoints',
                                    args.eval_split + '_' + suff + '_gencaps.pkl')
        print (results_file)
        pickle.dump(results_dict, open(results_file, 'wb'))

        print ("Number of samples with excessive repetitions:", n_rep)


if __name__ == '__main__':
    args = get_parser()
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    random.seed(1234)
    np.random.seed(1234)
    main(args)
