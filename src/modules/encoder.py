# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from torchvision.models import resnet18, resnet50, resnet101, resnet152, vgg16, vgg19, inception_v3
import torch
import torch.nn as nn
import random
import numpy as np


class EncoderCNN(nn.Module):
    def __init__(self, embed_size, dropout=0.5, image_model='resnet101', pretrained=True):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = globals()[image_model](pretrained=pretrained)
        modules = list(resnet.children())[:-2]  # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)

        self.linear = nn.Sequential(nn.Conv2d(resnet.fc.in_features, embed_size, kernel_size=1, padding=0),
                                    nn.Dropout2d(dropout))

    def forward(self, images, keep_cnn_gradients=False):
        """Extract feature vectors from input images."""

        if keep_cnn_gradients:
            raw_conv_feats = self.resnet(images)
        else:
            with torch.no_grad():
                raw_conv_feats = self.resnet(images)
        features = self.linear(raw_conv_feats)
        features = features.view(features.size(0), features.size(1), -1)

        return features


class EncoderLabels(nn.Module):
    def __init__(self, embed_size, num_classes, dropout=0.5, embed_weights=None, scale_grad=False):

        super(EncoderLabels, self).__init__()
        embeddinglayer = nn.Embedding(num_classes, embed_size, padding_idx=num_classes-1, scale_grad_by_freq=scale_grad)
        if embed_weights is not None:
            embeddinglayer.weight.data.copy_(embed_weights)
        self.pad_value = num_classes - 1
        self.linear = embeddinglayer
        self.dropout = dropout
        self.embed_size = embed_size

    def forward(self, x, onehot_flag=False):

        if onehot_flag:
            embeddings = torch.matmul(x, self.linear.weight)
        else:
            embeddings = self.linear(x)

        embeddings = nn.functional.dropout(embeddings, p=self.dropout, training=self.training)
        embeddings = embeddings.permute(0, 2, 1).contiguous()

        return embeddings
