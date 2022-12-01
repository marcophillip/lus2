#!/bin/bash
python train.py models=vgg19 params.lr=1e-5 params.epochs=2000 params.batch_size=64 models.trainable=False