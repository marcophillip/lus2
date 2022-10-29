import tensorflow as tf
from tensorflow.keras.layers import Conv2D,BatchNormalization,ReLU,Dense,PReLU,GlobalAveragePooling2D

class ChannelAttention(tf.keras.layers.Layer):

    
    def __init__(self, filters:int=1280, ratio:int=4):
        super(ChannelAttention, self).__init__()
        self.filters = filters
        self.ratio = ratio

    def build(self, input_shape):
        self.shared_layer_one = tf.keras.layers.Dense(self.filters//self.ratio,
                          activation='relu', kernel_initializer='he_normal', 
                          use_bias=True, 
                          bias_initializer='zeros')
        self.shared_layer_two = tf.keras.layers.Dense(self.filters,
                          kernel_initializer='he_normal',
                          use_bias=True,
                          bias_initializer='zeros')

    def call(self, inputs):
        # AvgPool
        avg_pool = tf.keras.layers.GlobalAveragePooling2D()(inputs)
        

        avg_pool = self.shared_layer_one(avg_pool)
        avg_pool = self.shared_layer_two(avg_pool)

        # MaxPool
        max_pool = tf.keras.layers.GlobalMaxPooling2D()(inputs)
        max_pool = tf.keras.layers.Reshape((1,1,self.filters))(max_pool)

        max_pool = self.shared_layer_one(max_pool)
        max_pool = self.shared_layer_two(max_pool)


        attention = tf.keras.layers.Add()([avg_pool,max_pool])
        attention = tf.keras.layers.Activation('sigmoid')(attention)
        
        # return tf.keras.layers.Multiply()([inputs, attention])
        return attention




class SpatialAttention(tf.keras.layers.Layer):
    
    def __init__(self, kernel_size:int=7):
        super(SpatialAttention, self).__init__()
        self.kernel_size = kernel_size
  
    def build(self, input_shape):
        self.conv2d = tf.keras.layers.Conv2D(filters = 1,
                kernel_size=self.kernel_size,
                strides=1,
                padding='same',
                activation='sigmoid',
                kernel_initializer='he_normal',
                use_bias=False)

    def call(self, inputs):
        
        # AvgPool
        avg_pool = tf.keras.layers.Lambda(lambda x: tf.keras.backend.mean(x, axis=3, keepdims=True))(inputs)
        
        # MaxPool
        max_pool = tf.keras.layers.Lambda(lambda x: tf.keras.backend.max(x, axis=3, keepdims=True))(inputs)

        attention = tf.keras.layers.Concatenate(axis=3)([avg_pool, max_pool])

        attention = self.conv2d(attention)


        # return tf.keras.layers.multiply([inputs, attention]) 
        return attention



class SelfAttention(tf.keras.layers.Layer):
    def __init__(self):
        super(SelfAttention,self).__init__()

    def build(self,input_shape):
        n,self.h_,self.w_,c = input_shape
        self.hw = self.h_*self.w_
        self.conv1 = Conv2D(c//8,1,padding='same')
        self.conv2 = Conv2D(c//8,1,padding='same')
        self.conv3 = Conv2D(c//2,1,padding='same')
        self.conv4 = Conv2D(c,1,padding='same')
        self.gamma = self.add_weight(shape=[1],initializer='zeros',trainable=True)

    def call(self,x):
        theta = self.conv1(x)
        theta = tf.reshape(theta,(-1,self.n_feats,theta.shape[-1]))

        phi = self.conv2(x)
        phi = tf.nn.max_pool1d(phi,ksize=2,strides=2,padding='valid')
        phi = tf.reshape(phi,(-1,(self.n_feats//4,phi.shape[-1])))

        attn = tf.matmul(theta,phi,transpose_b=True)
        attn = tf.nn.softmax(attn)

        g = self.conv3(x)
        g = tf.nn.max_pool2d(g,ksize=2,strides=2,padding='valid')
        g = tf.reshape(g,(-1,self.n_feats//4,g.shape[-1]))

        attn_g = tf.matmul(attn,g)
        attn_g = tf.reshape(attn_g,(-1,self.h_,self.w_,attn_g.shape[-1]))
        attn_g = self.conv4(attn_g)

        out = x+self.gamma*attn_g
        return out


class MultiAtrousConv(tf.keras.layers.Layer):
    def __init__(self,filters):
        super(MultiAtrousConv,self).__init__()

        self.d0 = Conv2D(512,3,dilation_rate=6,padding='same')
        self.d1 = Conv2D(512,3,dilation_rate=12,padding='same')
        self.d2 = Conv2D(512,3,dilation_rate=18,padding='same')

        self.conv = Conv2D(filters,1)
        
    def call(self,x):
        x0 = self.d0(x)
        x1 = self.d1(x)
        x2 = self.d2(x)
        x = tf.keras.layers.concatenate([x0,x1,x2],axis=3)
        x = self.conv(x)
        x = tf.nn.relu(x)
        return x

class GeMPoolingLayer(tf.keras.layers.Layer):
    def __init__(self, p=1., train_p=False):
        super().__init__()
        if train_p:
            self.p = tf.Variable(p, dtype=tf.float32)
        else:
            self.p = p
        self.eps = 1e-6

    def call(self, inputs: tf.Tensor, **kwargs):
        inputs = tf.clip_by_value(inputs, clip_value_min=1e-6, clip_value_max=tf.reduce_max(inputs))
        inputs = tf.pow(inputs, self.p)
        inputs = tf.reduce_mean(inputs, axis=[1, 2], keepdims=False)
        inputs = tf.pow(inputs, 1./self.p)
        return inputs



class SpatialAttention2D(tf.keras.layers.Layer):
    def __init__(self,filters):
        super(SpatialAttention2D,self).__init__()
        self.conv1 = Conv2D(filters,1,1)
        self.bn = BatchNormalization()
        self.act1 = ReLU()
        self.conv2 = Conv2D(filters,1,1)

    # def build(self,input_shape):
    #     _,_,_,c = input_shape

    def call(self,x):
        x = self.conv1(x)
        x = self.bn(x)

        feat_map_norm = tf.math.l2_normalize(x)

        x = self.act1(x)
        x = self.conv2(x)
        att_score =tf.keras.activations.softplus(x)
       
        # att = tf.reshape(att_score,feat_map_norm.shape)

        x = att_score*feat_map_norm
        return x,att_score



class OrthogonalFusion(tf.keras.layers.Layer):
    def __init__(self):
        super(OrthogonalFusion,self).__init__()

    
    def call(self,x):
      
        fl,fg = x
    
        bs,h,w,c = fl.shape

        print(fl,fg)

        print(tf.reshape(fl,[bs,h*w,c]))



        fl_dot_fg =tf.matmul(fg[:,None,:],tf.reshape(fl,[bs,h*w,c]))
        print(fl_dot_fg)
        fl_dot_fg = tf.reshape(fl_dot_fg,(bs,h,w,1))
        fg_norm = tf.math.l2_normalize(fg)

        fl_proj = (fl_dot_fg/fg_norm[:,None,None,None])*fg[:,:,None,None]
        fl_orth = fl-fl_proj

        f_fused = tf.concat([fl_orth,tf.repeat(fg[:,:,None,None],repeats=(1,1,w,h),axis=1)])
        return f_fused


class dolg_head(tf.keras.layers.Layer):
    def __init__(self):
        super(dolg_head,self).__init__()
        feature_dim_l_g = 512
        # fusion_out = 2*feature_dim_l_g

        self.gem = GeMPoolingLayer()
        
        self.head = tf.keras.Sequential([
                                    GlobalAveragePooling2D(),
                                       Dense(512),
                                      BatchNormalization(),
                                      PReLU(),
                                      Dense(4,activation='sigmoid'),
                                        ])

        self.mam = MultiAtrousConv(512)
        self.conv_g=Conv2D(feature_dim_l_g,1)
        self.bn_g = BatchNormalization()
        self.act_g = tf.keras.layers.ReLU()

        self.attention = SpatialAttention2D(feature_dim_l_g)
        self.fusion=OrthogonalFusion()

    def call(self,x):

        x_l,x_g = x

        x_l = self.mam(x_l)
    
        x_l,attn_score = self.attention(x_l)

        x_g = self.conv_g(x_g)
        x_g = self.bn_g(x_g)
        x_g = self.act_g(x_g)

        x_g = self.gem(x_g)
        # x_g = x_g[:,None,None,:]
  
        # x_fused =self.fusion([x_l,x_g])

        ############################33
        

        fl,fg = x_l,x_g
        bs,h,w,c = fl.shape

        print(fl.dtype,fg)

        print(tf.reshape(fl,[bs,h*w,c]))



        fl_dot_fg =tf.matmul(fg[:,None,:],tf.reshape(fl,[bs,h*w,c]))
        print(fl_dot_fg)
        fl_dot_fg = tf.reshape(fl_dot_fg,(bs,h,w,1))
        fg_norm = tf.math.l2_normalize(fg)

        fl_proj = (fl_dot_fg/fg_norm[:,None,None,None])*fg[:,:,None,None]
        fl_orth = fl-fl_proj

        f_fused = tf.concat([fl_orth,tf.repeat(fg[:,:,None,None],repeats=(1,1,w,h),axis=1)])
        return f_fused

    


                                      

