import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from src.data.dataset import TrainGenerator,ValGenerator
import tensorflow as tf
from src.models import Eff
import hydra
from hydra.core.config_store import ConfigStore
from src.config import Config
import os
import wandb
from wandb.keras import WandbCallback
from omegaconf import DictConfig, OmegaConf
from datetime import datetime
from src.losses import focal_loss
wandb.init(project="lus-artifacts", entity="marcphillip")


pwd_ = '/home/g007markphillip/model_artifacts2'

@hydra.main(version_base=None,config_path='conf',config_name='config')
def main(cfg:Config):
    print(OmegaConf.to_yaml(cfg))
    
    wandb.config=cfg

    checkpoint_filepath = os.path.join(pwd_,cfg.paths.weights,cfg.models.model_name,datetime.now().strftime('%Y-%m-%d-%H:%M:%S'),"cp-{epoch:04d}.ckpt")
    
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='loss',
        mode='min',
        save_freq =50*cfg.params.batch_size,
    )
    

    tensorboard_cp=tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join(pwd_,cfg.paths.logs,cfg.models.model_name),
        update_freq='epoch',
    )

    # reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',verbose=1, factor=0.01,
    #                           patience=5)

    
    model = Eff(cfg.models.model_name,
                cfg.models.trainable,
                # cfg.models.unfreeze_layers
                )
#    print(model.build_graph()[0].summary())
    # latest=tf.train.latest_checkpoint('models/weights/efficientnetB0_noisystudent/2022-10-06-22:49:00')
    
#   lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#         0.0001,
#         decay_steps=400,
#         decay_rate=0.096,
#         staircase=True)

    def scheduler(epoch,lr):
        if epoch>600:lr=1e-7
        return lr

    scheduler_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

    model.compile(
    optimizer=tf.keras.optimizers.Adam(cfg.params.lr),
    loss = focal_loss(),
    metrics=[
        # tf.keras.metrics.BinaryAccuracy(threshold=0.8)
        tf.keras.metrics.AUC()
        ]
              )
    
    train_data = TrainGenerator(2).batch(cfg.params.batch_size)
    validation_data = ValGenerator(2).batch(cfg.params.batch_size)

  

    model.fit(train_data,
              validation_data=validation_data,
              epochs=cfg.params.epochs,
              callbacks=[
                       model_checkpoint_callback,
                    #    scheduler_callback,
                       tensorboard_cp,
                       WandbCallback(save_weights_only=False,
                                     save_model=False,
                                     predictions=20)
                         ]
                      )          


if __name__ == '__main__':
    main()
    
                
