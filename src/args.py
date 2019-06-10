# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import argparse
import os


def get_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('--save_dir', type=str, default='path/to/save/models',
                        help='path where checkpoints will be saved')

    parser.add_argument('--project_name', type=str, default='inversecooking',
                        help='name of the directory where models will be saved within save_dir')

    parser.add_argument('--model_name', type=str, default='model',
                        help='save_dir/project_name/model_name will be the path where logs and checkpoints are stored')

    parser.add_argument('--transfer_from', type=str, default='',
                        help='specify model name to transfer from')

    parser.add_argument('--suff', type=str, default='',
                        help='the id of the dictionary to load for training')

    parser.add_argument('--image_model', type=str, default='resnet50', choices=['resnet18', 'resnet50', 'resnet101',
                                                                                 'resnet152', 'inception_v3'])

    parser.add_argument('--recipe1m_dir', type=str, default='path/to/recipe1m',
                        help='directory where recipe1m dataset is extracted')

    parser.add_argument('--aux_data_dir', type=str, default='../data',
                        help='path to other necessary data files (eg. vocabularies)')

    parser.add_argument('--crop_size', type=int, default=224, help='size for randomly or center cropping images')

    parser.add_argument('--image_size', type=int, default=256, help='size to rescale images')

    parser.add_argument('--log_step', type=int , default=10, help='step size for printing log info')

    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='base learning rate')

    parser.add_argument('--scale_learning_rate_cnn', type=float, default=0.01,
                        help='lr multiplier for cnn weights')

    parser.add_argument('--lr_decay_rate', type=float, default=0.99,
                        help='learning rate decay factor')

    parser.add_argument('--lr_decay_every', type=int, default=1,
                        help='frequency of learning rate decay (default is every epoch)')

    parser.add_argument('--weight_decay', type=float, default=0.)

    parser.add_argument('--embed_size', type=int, default=512,
                        help='hidden size for all projections')

    parser.add_argument('--n_att', type=int, default=8,
                        help='number of attention heads in the instruction decoder')

    parser.add_argument('--n_att_ingrs', type=int, default=4,
                        help='number of attention heads in the ingredient decoder')

    parser.add_argument('--transf_layers', type=int, default=16,
                        help='number of transformer layers in the instruction decoder')

    parser.add_argument('--transf_layers_ingrs', type=int, default=4,
                        help='number of transformer layers in the ingredient decoder')

    parser.add_argument('--num_epochs', type=int, default=400,
                        help='maximum number of epochs')

    parser.add_argument('--batch_size', type=int, default=128)

    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--dropout_encoder', type=float, default=0.3,
                        help='dropout ratio for the image and ingredient encoders')

    parser.add_argument('--dropout_decoder_r', type=float, default=0.3,
                        help='dropout ratio in the instruction decoder')

    parser.add_argument('--dropout_decoder_i', type=float, default=0.3,
                        help='dropout ratio in the ingredient decoder')

    parser.add_argument('--finetune_after', type=int, default=-1,
                        help='epoch to start training cnn. -1 is never, 0 is from the beginning')

    parser.add_argument('--loss_weight', nargs='+', type=float, default=[1.0, 0.0, 0.0, 0.0],
                        help='training loss weights. 1) instruction, 2) ingredient, 3) eos 4) cardinality')

    parser.add_argument('--max_eval', type=int, default=4096,
                        help='number of validation samples to evaluate during training')

    parser.add_argument('--label_smoothing_ingr', type=float, default=0.1,
                        help='label smoothing for bce loss for ingredients')

    parser.add_argument('--patience', type=int, default=50,
                        help='maximum number of epochs to allow before early stopping')

    parser.add_argument('--maxseqlen', type=int, default=15,
                        help='maximum length of each instruction')

    parser.add_argument('--maxnuminstrs', type=int, default=10,
                        help='maximum number of instructions')

    parser.add_argument('--maxnumims', type=int, default=5,
                        help='maximum number of images per sample')

    parser.add_argument('--maxnumlabels', type=int, default=20,
                        help='maximum number of ingredients per sample')

    parser.add_argument('--es_metric', type=str, default='loss', choices=['loss', 'iou_sample'],
                        help='early stopping metric to track')

    parser.add_argument('--eval_split', type=str, default='val')

    parser.add_argument('--numgens', type=int, default=3)

    parser.add_argument('--greedy', dest='greedy', action='store_true',
                        help='enables greedy sampling (inference only)')
    parser.set_defaults(greedy=False)

    parser.add_argument('--temperature', type=float, default=1.0,
                        help='sampling temperature (when greedy is False)')

    parser.add_argument('--beam', type=int, default=-1,
                        help='beam size. -1 means no beam search (either greedy or sampling)')

    parser.add_argument('--ingrs_only', dest='ingrs_only', action='store_true',
                        help='train or evaluate the model only for ingredient prediction')
    parser.set_defaults(ingrs_only=False)

    parser.add_argument('--recipe_only', dest='recipe_only', action='store_true',
                        help='train or evaluate the model only for instruction generation')
    parser.set_defaults(recipe_only=False)

    parser.add_argument('--log_term', dest='log_term', action='store_true',
                        help='if used, shows training log in stdout instead of saving it to a file.')
    parser.set_defaults(log_term=False)

    parser.add_argument('--notensorboard', dest='tensorboard', action='store_false',
                        help='if used, tensorboard logs will not be saved')
    parser.set_defaults(tensorboard=True)

    parser.add_argument('--resume', dest='resume', action='store_true',
                        help='resume training from the checkpoint in model_name')
    parser.set_defaults(resume=False)

    parser.add_argument('--nodecay_lr', dest='decay_lr', action='store_false',
                        help='disables learning rate decay')
    parser.set_defaults(decay_lr=True)

    parser.add_argument('--load_jpeg', dest='use_lmdb', action='store_false',
                        help='if used, images are loaded from jpg files instead of lmdb')
    parser.set_defaults(use_lmdb=True)

    parser.add_argument('--get_perplexity', dest='get_perplexity', action='store_true',
                        help='used to get perplexity in evaluation')
    parser.set_defaults(get_perplexity=False)

    parser.add_argument('--use_true_ingrs', dest='use_true_ingrs', action='store_true',
                        help='if used, true ingredients will be used as input to obtain the recipe in evaluation')
    parser.set_defaults(use_true_ingrs=False)

    args = parser.parse_args()

    return args
