class Config: 
    PRIOR_TYPE = 'logit_normal'
    DIVERGENCE_TYPE = 'JS'
    USE_SQRT_PENALTY = False  

    EXP_PREFIX = f"{'sqrt_' if USE_SQRT_PENALTY else ''}{DIVERGENCE_TYPE}_{PRIOR_TYPE}"
    DATA_DIR = '/root/wae/datasets/img_align_celeba'
    LOG_DIR = '/root/wae/logs'

    EPOCHS = 500
    DISCRIMINATOR_STEPS = 3
    AUX_STEPS = 3
    BATCH_SIZE = 512
    LR = 8e-4
    CHANNELS = 3

    IMAGE_SIZE = 64
    LATENT_DIM = 64
    HIDDEN_DIM = 64

    LAMBDA = 100.0   

    ACCELERATOR = "gpu"
    DEVICES = [0, 1] 
    STRATEGY = "ddp_find_unused_parameters_true"
    NUM_WORKERS = 4
    SEED = 123

    @classmethod
    def to_dict(cls):
        return {
            k: v for k, v in cls.__dict__.items() 
            if not k.startswith('__') 
            and not callable(v) 
            and not isinstance(v, (classmethod, staticmethod))
        }