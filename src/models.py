import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.layers import Average, GlobalAveragePooling2D,Dense,Dropout
from efficientnet import tfkeras as efficientnet
from src.layers import SpatialAttention, ChannelAttention


class Eff(tf.keras.models.Model):
    def __init__(self,model_name:str,trainable:bool,unfreeze_layers:int=None):
        super(Eff,self).__init__()
        
        models = {
                #    'resnet50v2':tf.keras.applications.ResNet50V2(
                #                       include_top=False,
                #                       weights="imagenet",
                #                       input_shape=(224,224, 3)),
            
                   'vgg19': tf.keras.applications.VGG19(
                                          include_top=False,
                                          weights='imagenet',
                                          input_shape=(224,224,3)
                                                 ),
                   'efficientnetB0_noisystudent':efficientnet.EfficientNetB0(
                                      include_top=False,
                                    #   pooling='avg',
                                      weights="noisy-student",
                                      input_shape=(300,300, 3)),

                    'efficientnetB0_imagenet':tf.keras.applications.EfficientNetB0(
                                      include_top=False,
                                      weights='imagenet',
                                      input_shape=(224,224, 3)
                                        ),
            
                   'efficientnetB1_noisystudent':efficientnet.EfficientNetB1(
                                      include_top=False,
#                                       pooling='avg',
                                      weights="noisy-student",
                                      input_shape=(240,240, 3)),
            
                    'efficientnetB1_imagenet':tf.keras.applications.EfficientNetB1(
                                      include_top=False,
                                      weights="imagenet",
                                      input_shape=(240,240, 3)),
            
                   'efficientnetB2_noisystudent':efficientnet.EfficientNetB2(
                                      include_top=False,
#                                       pooling='avg',
                                      weights="noisy-student",
                                      input_shape=(260,260, 3)),
            
                    'efficientnetB2_imagenet':tf.keras.applications.EfficientNetB2(
                                      include_top=False,
                                      weights="imagenet",
                                      input_shape=(260,260, 3)),

                    'efficientnetB3_imagenet':tf.keras.applications.EfficientNetB3(
                                      include_top=False,
                                      weights="imagenet",
                                      input_shape=(300,300, 3)),

                    'efficientnetB3_noisystudent':efficientnet.EfficientNetB3(
                                      include_top=False,
                                      pooling='avg',
                                      weights="noisy-student",
                                      input_shape=(300,300, 3)),
                }

        self.backbone = models[model_name]
        self.backbone.trainable=trainable

        self.can_module   = ChannelAttention()
        self.san_module_x = SpatialAttention()
        self.san_module_y = SpatialAttention()

        if unfreeze_layers is not None:
            

            for layer in self.backbone.layers[-unfreeze_layers:]:
                if not isinstance(layer, tf.keras.layers.BatchNormalization):
                    layer.trainable = True
            for layer in self.backbone.layers[-unfreeze_layers-1:]:
                print("{}---{}".format(layer.name,layer.trainable))


        self.out=tf.keras.layers.Dense(4, activation='sigmoid')


        
    def call(self,inputs):
        base_out = self.backbone(inputs)
  

        canx   = self.can_module(base_out)*base_out
        spnx   = self.san_module_x(canx)*canx
        spny   = self.san_module_y(canx)
        
        gapx   = GlobalAveragePooling2D()(spnx)
        wvgx   = GlobalAveragePooling2D()(spny)
        gapavg = Average()([gapx, wvgx])

        
        x = self.out(gapavg)
   
        return  x

    def build_graph(self):
        x = tf.keras.Input(shape=(224,224,3))
        return [self.backbone,tf.keras.Model(inputs=[x], outputs=self.call(x))]





