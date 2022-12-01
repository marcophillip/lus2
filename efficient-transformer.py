#!/usr/bin/env python
# coding: utf-8
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
sys.path.append('/home/g007markphillip/model_artifacts2/')
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
from datetime import datetime
import albumentations as A
import random
import numpy as np
from src.losses import focal_loss
import wandb
from wandb.keras import WandbCallback
wandb.init(project="lus-artifacts", entity="marcphillip")

tf.config.run_functions_eagerly(True)

m = tf.keras.applications.EfficientNetB0(weights=None,
        include_top=False, input_shape=[224,224, 3]
    )


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = tf.keras.layers.Dense(units, activation=tf.nn.gelu)(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    return x





class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self,filters):
        super(TransformerEncoder,self).__init__()
        self.filters = filters
        self.attn = tf.keras.layers.MultiHeadAttention(4,key_dim=filters,dropout=0.2)

        self.ln1 = tf.keras.layers.LayerNormalization()
        self.add1 = tf.keras.layers.Add()
        self.ln2 = tf.keras.layers.LayerNormalization()
        self.add2 =tf.keras.layers.Add()
        self.dense1 = tf.keras.layers.Dense(filters)
        self.dense2 = tf.keras.layers.Dense(filters)
        
    def call(self,x):
        x1 = self.ln1(x)
        x1 = self.attn(x1,x1)
        x2 = self.add1([x,x1])
        x3 = self.ln2(x2)
        x3 = mlp(x3,[self.filters],0.4)
        return x3
        





class Encoder(tf.keras.layers.Layer):
    def __init__(self,backbone):
        super(Encoder,self).__init__()
        self.backbone = backbone
        self.transformer_encoder = TransformerEncoder(1280)
        
        self.label_embedding = tf.keras.layers.Embedding(4,1280)
        self.state_embedding  =tf.keras.layers.Embedding(3,1280)
        self.concat = tf.keras.layers.Concatenate(1)
        self.out = tf.keras.layers.Dense(4)
        
    def call(self,image,states):
#         print(f"image {image}")
        constant_labels = tf.constant([0,1,2,3])
#         print(f"constant_labels {constant_labels}")
        label_embeddings = self.label_embedding(constant_labels)
#         print(f"label_embeddings {label_embeddings}")
        state_embeddings = self.state_embedding(states)
#         print(f"state_embeddings {state_embeddings}")
        
        label_embeddings+=state_embeddings
#         print(f"label_embeddings {label_embeddings}")
        features = self.backbone(image)
#         print(f"features {features}")
       
        features = tf.reshape(features,(-1,features.shape[1]*features.shape[2],features.shape[-1]))
#         print(f"features {features}")
        embeddings = self.concat([features,label_embeddings])
#         print(f"embeddings {embeddings}")
        embeddings = self.transformer_encoder(embeddings)
        
        label_embeddings = embeddings[:,-4:,:]
#         print(f"label_embeddings {label_embeddings}")
        out = self.out(label_embeddings)
#         print(f"out--{out}")
        diag_mask =  tf.tile(tf.eye(4),(out.shape[0],1))
#         print(f"diag_mask--{diag_mask}")
        out = out*tf.reshape(diag_mask,tf.shape(out))
        out = tf.reduce_sum(out,1)
#         print(f"out2 {out}")
        return tf.nn.sigmoid(out)
        


pwd_ = '/home/g007markphillip/model_artifacts2'
image_data_file=os.path.join(pwd_,'data','processed','files','balanced_processed.csv')
images_paths = os.path.join(pwd_,'data','processed','images')

data=pd.read_csv(image_data_file)
data.image=data.image.apply(lambda x: os.path.join(images_paths,x))

transform = A.Compose([
    # A.Rotate(limit=10,border_mode=0,p=0.5),
#     A.Affine(translate_percent=0.15),

    A.HorizontalFlip(p=0.3),
    A.VerticalFlip(p=0.3),

#     A.RandomBrightness(limit=0.2, p=0.5),
#     A.RandomContrast(limit=0.2, p=0.5),

    # A.OneOf([
    #     A.OpticalDistortion(distort_limit=1.),
    #     A.GridDistortion(num_steps=5, distort_limit=1.),
    # ], p=0.5),

    # A.HueSaturationValue(hue_shift_limit=40, sat_shift_limit=40, val_shift_limit=0, p=0.75),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, border_mode=0, p=0.6),
#     A.CLAHE (clip_limit=4.0, tile_grid_size=(2, 2), always_apply=False, p=1.0)
])




class TrainGenerator(tf.data.Dataset):
    
    def generator(fold):
        train = data[data.group_kfold.isin([0,1,2])]
        image_paths = train.image.values.tolist()
        classes = train[['alines', 'blines','consolidation','effusion']].values.tolist()

        i=0
        while i<len(train):
#             try:
            image = cv2.imread(image_paths[i])
#                 image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)


            image= cv2.resize(image,(224,224))
            image = transform(image=image)['image']

            image = tf.keras.applications.efficientnet.preprocess_input(image)

            num_known=random.randint(0,int(4*0.75))
            unk_indices =random.sample(range(4),(4-num_known))
           
#             print(unk_indices)

            class_ = np.array(classes[i])
#             print(class_)
            states = np.where(class_==1,1,2)      #positive state 1
                                                  #negative state 2

            states[unk_indices]=0    #uce unknown tokens

            loss_masks = np.where(states==0,1,0)
#             print(loss_masks)

            yield image, class_, states,loss_masks

            i+=1
#             except:
#                 i+=1
            
    def __new__(cls,fold):
                    
        return tf.data.Dataset.from_generator(cls.generator,
                                              output_signature=(
                                                         tf.TensorSpec(shape=(224,224,3), dtype=tf.float32),
                                                         tf.TensorSpec(shape=(4), dtype=tf.float32),
                                                         tf.TensorSpec(shape=(4), dtype=tf.float32),
                                                         tf.TensorSpec(shape=(4), dtype=tf.float32),
                                              ),
                                               
                                              args=(fold,)
                                             )
    
class ValGenerator(tf.data.Dataset):
    
    def generator(fold):
        train = data[data.group_kfold.isin([3])]
        image_paths = train.image.values.tolist()
        classes = train[['alines', 'blines','consolidation','effusion']].values.tolist()

        i=0
        while i<len(train):
#             try:
            image = cv2.imread(image_paths[i])
#                 image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)


            image= cv2.resize(image,(224,224))
        
            image = tf.keras.applications.efficientnet.preprocess_input(image)

            num_known=random.randint(0,int(4*0.75))
            unk_indices =random.sample(range(4),(4-num_known))
           
#             print(unk_indices)

            class_ = np.array(classes[i])
#             print(class_)
            states = np.where(class_==1,1,2)      #positive state 1
                                                  #negative state 2

            states[unk_indices]=0    ##uce unknown tokens

            loss_masks = np.where(states==0,1,0)
#             print(loss_masks)

            yield image, class_, states,loss_masks

            i+=1
#             except:
#                 i+=1
            
    def __new__(cls,fold):
                    
        return tf.data.Dataset.from_generator(cls.generator,
                                              output_signature=(
                                                         tf.TensorSpec(shape=(224,224,3), dtype=tf.float32),
                                                         tf.TensorSpec(shape=(4), dtype=tf.float32),
                                                         tf.TensorSpec(shape=(4), dtype=tf.float32),
                                                         tf.TensorSpec(shape=(4), dtype=tf.float32),
                                              ),
                                               
                                              args=(fold,)
                                             )



class Trainer(tf.keras.Model):
    def __init__(self,backbone):
        super(Trainer,self).__init__()
        backbone.trainable=False
        self.encoder  = Encoder(backbone)
               
        
    @property    
    def metrics(self):
        return [self.loss_tracker,self.acc,self.val_loss_tracker,self.test_acc]
    
    def compile(self,optimizer):
        super(Trainer,self).compile()
        self.optimizer = optimizer
        self.loss_fn = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        
        self.acc=tf.keras.metrics.BinaryAccuracy(
                name='train_binary_accuracy', dtype=None, threshold=0.7
        )
        self.test_acc=tf.keras.metrics.BinaryAccuracy(
                        name='val_binary_accuracy', dtype=None, threshold=0.7
                )
        self.loss_tracker = tf.keras.metrics.Mean(name='train-loss')
        self.val_loss_tracker = tf.keras.metrics.Mean(name='val-loss')
        
    @tf.function
    def train_step(self,data):
        images,labels,states,loss_masks = data

        with tf.GradientTape() as tape:
            y_pred = self.encoder(images,states)
            
            y_pred = tf.expand_dims(y_pred,-1)
            labels = tf.expand_dims(labels,-1)
                   
            loss = self.loss_fn(labels,y_pred)
            
            loss = loss*loss_masks
            loss = tf.reduce_sum(tf.reduce_mean(loss,-1))
        
        grads = tape.gradient(loss,self.encoder.trainable_variables)  
        self.optimizer.apply_gradients(zip(grads,self.encoder.trainable_variables))
        
        self.loss_tracker.update_state(loss)
        self.acc.update_state(labels,y_pred)
        return {'loss': self.loss_tracker.result(),'acc':self.acc.result()}

    @tf.function
    def test_step(self,data):
        images,labels,states,_= data
        new_states = tf.zeros_like(states)
        y_pred = self.encoder(images,new_states)
        loss = self.loss_fn(labels,y_pred)
        loss = tf.reduce_sum(tf.reduce_mean(loss,-1))
        
        self.val_loss_tracker.update_state(loss)
        self.test_acc.update_state(labels,y_pred)
        return {'val_loss':self.val_loss_tracker.result(),'val_acc':self.test_acc.result()}
        
        
        




model = Trainer(m)
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.000001))




train_data = TrainGenerator(2).batch(32)
val_data = ValGenerator(2).batch(32)

checkpoint_filepath = os.path.join(pwd_,'models','efficient-transformer',datetime.now().strftime('%Y-%m-%d-%H:%M:%S'),"cp-{epoch:04d}.ckpt")
    
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='loss',
    mode='min',
    save_freq =5*32,
)

model.fit(train_data,
          validation_data=val_data,
          callbacks=[model_checkpoint_callback,
                    WandbCallback(save_weights_only=False,
                                    save_model=False,
                                    predictions=20)
                                               ],
         epochs=2000)

