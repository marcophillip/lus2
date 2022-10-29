from dataclasses import dataclass

@dataclass
class Params:
    lr: float
    epochs: int
    batch_size: int

@dataclass
class Paths:
   logs: str
   weights: str

@dataclass
class Models:
    model_name: str
    image_size: int
    trainable: bool
    unfreeze_layers: int
@dataclass
class Config:
    params: Params
    paths: Paths
    models: Models
