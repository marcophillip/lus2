#!/bin/bash
python train.py params.lr=1e-5 params.epochs=2000 params.batch_size=16 models.trainable=True
