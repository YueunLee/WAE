import pytorch_lightning as pl 
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor 
from pytorch_lightning.loggers import TensorBoardLogger 
from config import Config 
from trainer import WAE_GAN 
from data_module import CelebADataModule 
import torch
import os
import json
from datetime import datetime
import torch.multiprocessing as mp

torch.set_float32_matmul_precision("high")
os.environ["NCCL_P2P_DISABLE"] = "1"

def main(): 
    global_rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    config_dict = Config.to_dict()
    
    if "WAE_RUN_TIMESTAMP" not in os.environ:
        os.environ["WAE_RUN_TIMESTAMP"] = datetime.now().strftime("%m%d_%H%M%S")
    
    current_time = os.environ["WAE_RUN_TIMESTAMP"]

    exp_name = f"{config_dict['EXP_PREFIX']}_{current_time}"
    config_dict['EXP_NAME'] = exp_name
    
    log_dir_path = os.path.join(config_dict['LOG_DIR'], exp_name)
    
    pl.seed_everything(config_dict['SEED'])

    if global_rank == 0 and local_rank == 0:
        os.makedirs(log_dir_path, exist_ok=True)
        config_path = os.path.join(log_dir_path, "config.json")
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=4)
        print(f"Configuration saved to: {config_path}")

    model = WAE_GAN(config_dict)
    dm = CelebADataModule(config_dict)

    tb_logger = TensorBoardLogger(save_dir=config_dict['LOG_DIR'], name=exp_name)

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{log_dir_path}/checkpoints",
        filename='best-fid-{epoch:02d}-{val_fid:.2f}',
        monitor='val_fid',      
        mode='min',             
        save_top_k=3,           
        save_last=True,       
        verbose=(global_rank == 0),        
        save_weights_only=False 
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(
        accelerator=config_dict['ACCELERATOR'],
        devices=config_dict['DEVICES'],
        strategy=config_dict['STRATEGY'],
        max_epochs=config_dict['EPOCHS'],
        callbacks=[checkpoint_callback, lr_monitor],
        logger=tb_logger,
        precision="bf16-mixed",
        log_every_n_steps=10,
        check_val_every_n_epoch=5,
        num_sanity_val_steps=0
    )

    trainer.fit(model, datamodule=dm)

if __name__ == '__main__':
    try:
        if mp.get_start_method(allow_none=True) != 'spawn':
            mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()