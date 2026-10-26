import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
from stable_pretraining._config import get_config as get_spt_config
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import OmegaConf, open_dict

from module import SIGReg
from utils import get_column_normalizer, get_img_preprocessor, SaveCkptCallback


class RoutingStepCallback(pl.Callback):
    """Commit one globally aggregated load update per optimizer step.

    Counts may accumulate across microbatches. Validation and counterfactual
    forwards do not contribute counts. Checkpointed controller buffers preserve
    the load and bias history across resume and downstream evaluation.
    """

    def on_train_start(self, trainer, pl_module):
        self.last_step = int(trainer.global_step)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        current_step = int(trainer.global_step)
        if current_step == self.last_step:
            return
        routing = getattr(pl_module.model.predictor, "routing", None)
        if routing is not None:
            metrics = routing.after_step()
            if metrics:
                pl_module.log_dict(
                    {f"train/route_{name}": value for name, value in metrics.items()},
                    on_step=True, on_epoch=False, sync_dist=False,
                )
        self.last_step = current_step


def initialize_visual_encoder_from_checkpoint(
    model,
    checkpoint,
    *,
    load_projector=True,
    strict=True,
):
    """Load the visual latent coordinate system from a pretrained model."""
    checkpoint = Path(str(checkpoint)).expanduser()
    if checkpoint.exists():
        if checkpoint.is_dir():
            weight_files = sorted(checkpoint.glob("*.pt"))
            if len(weight_files) != 1:
                raise ValueError(
                    "pretrained visual checkpoint folders must contain "
                    "exactly one .pt file; provide an explicit file when "
                    "multiple checkpoints are present"
                )
            checkpoint = weight_files[0]
        if not checkpoint.is_file() or checkpoint.suffix != ".pt":
            raise ValueError(
                "pretrained visual checkpoint must be a .pt file or a "
                "folder containing one .pt file"
            )
        state_dict = torch.load(checkpoint, map_location="cpu")

        def component_state(name):
            prefixes = (f"{name}.", f"model.{name}.")
            for prefix in prefixes:
                selected = {
                    key[len(prefix) :]: value
                    for key, value in state_dict.items()
                    if key.startswith(prefix)
                }
                if selected:
                    return selected
            raise KeyError(
                f"checkpoint contains no parameters for {name}"
            )

        model.encoder.load_state_dict(
            component_state("encoder"),
            strict=bool(strict),
        )
        if load_projector:
            model.projector.load_state_dict(
                component_state("projector"),
                strict=bool(strict),
            )
    else:
        pretrained = swm.wm.utils.load_pretrained(str(checkpoint))
        model.encoder.load_state_dict(
            pretrained.encoder.state_dict(),
            strict=bool(strict),
        )
        if load_projector:
            model.projector.load_state_dict(
                pretrained.projector.state_dict(),
                strict=bool(strict),
            )
    return model


def flow_jepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.history_size
    n_preds = cfg.num_preds
    lambd = cfg.loss.sigreg.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, K, D)
    act_emb = output["act_emb"]

    # Loss
    ctx_emb = emb[:, :ctx_len]
    future_act = act_emb[:, ctx_len - 1 : ctx_len - 1 + n_preds]
    tgt_emb = emb[:, ctx_len : ctx_len + n_preds]
    losses = self.model.flow_loss(
        ctx_emb, future_act, tgt_emb,
        training_step=int(self.global_step), return_details=True,
    )
    output["pred_loss"] = losses["pred_loss"]
    output["route_credit_loss"] = losses["route_credit_loss"]
    sigreg_emb = emb.permute(1, 2, 0, 3).reshape(
        emb.size(1) * emb.size(2), emb.size(0), emb.size(3)
    )
    output["sigreg_loss"]= self.sigreg(sigreg_emb)
    credit_weight = float(self.model.routing_credit.get("weight", 0.01))
    output["loss"] = (output["pred_loss"] + lambd * output["sigreg_loss"]
                      + credit_weight * output["route_credit_loss"])

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, on_epoch=True, sync_dist=True,
                  batch_size=emb.size(0))
    if losses["routing_metrics"]:
        self.log_dict(
            {f"{stage}/{key}": value.detach()
             for key, value in losses["routing_metrics"].items()},
            on_step=True, on_epoch=True, sync_dist=True, batch_size=emb.size(0),
        )
    return output

@hydra.main(version_base=None, config_path="./config/train", config_name="fjepa")
def run(cfg):
    pl.seed_everything(int(cfg.seed), workers=True)
    # The installed SPT Manager otherwise replaces explicit CSV loggers and
    # redirects their files to its registry directory.
    spt.set(default_loggers={"registry": False})
    # Installed SPT's set(cache_dir=None) is a no-op, and an empty string is
    # interpreted as cwd. Explicit None selects its legacy managed-path mode.
    get_spt_config().cache_dir = None
    #########################
    ##       dataset       ##
    #########################

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)
    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]
    
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

        cfg.model.action_encoder.input_dim = (
            cfg.data.dataset.frameskip * dataset.get_dim("action")
        )

        pretrained_visual_cfg = cfg.get("pretrained_visual_encoder", {})
        pretrained_visual_enabled = bool(
            pretrained_visual_cfg.get("enabled", False)
        )
        if pretrained_visual_enabled:
            checkpoint = pretrained_visual_cfg.get("checkpoint", None)
            if not checkpoint:
                raise ValueError(
                    "pretrained_visual_encoder.checkpoint is required "
                    "when pretrained visual initialization is enabled"
                )
            freeze_visual = bool(
                pretrained_visual_cfg.get("freeze", True)
            )
            cfg.model.freeze_encoder = freeze_visual
            cfg.model.freeze_projector = (
                freeze_visual
                and bool(
                    pretrained_visual_cfg.get(
                        "load_projector",
                        True,
                    )
                )
            )

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(train_set, **cfg.loader,shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)
    
    ##############################
    ##       model / optim      ##
    ##############################

    world_model = hydra.utils.instantiate(cfg.model)
    pretrained_visual_cfg = cfg.get("pretrained_visual_encoder", {})
    if bool(pretrained_visual_cfg.get("enabled", False)):
        initialize_visual_encoder_from_checkpoint(
            world_model,
            pretrained_visual_cfg.checkpoint,
            load_projector=bool(
                pretrained_visual_cfg.get("load_projector", True)
            ),
            strict=bool(pretrained_visual_cfg.get("strict", True)),
        )
        world_model.set_visual_encoder_frozen(
            bool(cfg.model.freeze_encoder),
            bool(cfg.model.freeze_projector),
        )

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model = world_model,
        sigreg = SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(flow_jepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id)

    # Always retain locally readable metrics even when external logging is off.
    logger = [CSVLogger(save_dir=str(run_dir), name="metrics", version="",
                        flush_logs_every_n_steps=10)]
    if cfg.wandb.enabled:
        logger.append(WandbLogger(**cfg.wandb.config))
    for experiment_logger in logger:
        experiment_logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    process_rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    if process_rank == 0:
        with open(run_dir / "config.yaml", "w") as f:
            OmegaConf.save(cfg, f)

    object_dump_callback = SaveCkptCallback(
        run_name=cfg.output_model_name, cfg=cfg.model, epoch_interval=1,
    )
    resume_callback = ModelCheckpoint(
        dirpath=str(run_dir / "training_state"), filename="epoch-{epoch:02d}",
        save_last=True, save_top_k=1, every_n_epochs=1, save_weights_only=False,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback, RoutingStepCallback(), resume_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
        default_root_dir=str(run_dir),
    )

    ckpt_path = run_dir / "training_state" / "last.ckpt"
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        seed=int(cfg.seed),
        weights_only=False,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )

    manager()
    return


if __name__ == "__main__":
    run()
