import argparse
import json
import logging
import os
from tempfile import mkdtemp
import typing as tp
from argparse import ArgumentParser
from warnings import warn
# import hydra
import numpy as np
import math
from omegaconf import OmegaConf
import torch
import torch.nn as nn
import wandb
from matplotlib import pyplot as plt
from medAI.modeling.prostnfound import ProstNFound
from medAI.modeling.setr import SETR
from torch.nn import functional as F
from tqdm import tqdm
import copy
from torchvision.transforms import v2 as T
import time

from medAI.modeling import *
from medAI.utils.distributed import is_main_process
from medAI.utils.reproducibility import (
    get_all_rng_states,
    set_all_rng_states,
    set_global_seed,
)
from medAI.losses.prostnfound import (
    build_loss,
)
from medAI.layers.masked_prediction_module import (
    MaskedPredictionModule,
)
from projects.seg_ttt.dataloader_ttt import get_dataloaders_from_args
from medAI.factories.prostnfound.models import get_model
from medAI.engine.prostnfound.evaluator import (
    ProstNFoundEvaluator as Evaluator,
)
# from projects.seg_ttt.learn_ttt import SupervisedTrainerWithSegTTT
from projects.seg_ttt.TransUnet.seg_model import FrozenSegmentationModel
from projects.seg_ttt.TTA_baselines import (
    setup_tent, setup_eata, setup_sar, setup_memo, setup_roid,
    setup_cotta, setup_rotta, setup_petta, setup_rmt,
    compute_fishers_from_dict_loader
)
from projects.seg_ttt.TTA_baselines import SupervisedTrainerWithSegTTT
from medAI.modeling.proside import Proside
from projects.seg_ttt.bootstrap import compute_bootstrap_metrics

def main(cfg):

    if cfg.get("output_dir") is not None:
        os.makedirs(cfg.output_dir, exist_ok=True)
        OmegaConf.save(
            cfg, os.path.join(cfg.output_dir, "train_config.yaml"), resolve=True
        )

    # setup
    handlers = [logging.StreamHandler()]
    if cfg.output_dir is not None and is_main_process():
        file_handler = logging.FileHandler(os.path.join(cfg.output_dir, "training.log"))
        handlers.append(file_handler)

    logging.basicConfig(
        level=logging.INFO if not cfg.debug else logging.DEBUG,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
    )

    logging.info("Setting up experiment")
    wandb.init(
        config=OmegaConf.to_object(cfg),
        **cfg.get("wandb", {}),
    )
    _tmpdir = mkdtemp()
    OmegaConf.save(cfg, os.path.join(_tmpdir, "train_config.yaml"), resolve=True)
    wandb.save(
        os.path.join(_tmpdir, "train_config.yaml"), base_path=_tmpdir, policy="now"
    )
    # cfg.wandb_url = wandb.run.url if wandb.run else None

    def log_fn(data: dict):

        if wandb.run is not None:
            data_wandb = {}
            for k, v in data.items():
                if isinstance(v, plt.Figure):
                    data_wandb[k] = wandb.Image(v)
                else:
                    data_wandb[k] = v
            wandb.log(data_wandb)
        if cfg.get("output_dir") is not None:
            metrics_path = os.path.join(cfg.output_dir, "metrics.jsonl")

            figures = {k: v for k, v in data.items() if isinstance(v, plt.Figure)}
            scalars = {k: v for k, v in data.items() if not k in figures}

            with open(metrics_path, "a") as f:
                f.write(json.dumps(scalars) + "\n")

            for k, fig in figures.items():
                fig_dir = os.path.join(cfg.output_dir, "figures", k)
                os.makedirs(fig_dir, exist_ok=True)
                index = len(os.listdir(fig_dir))
                fig_path = os.path.join(fig_dir, f"{index:05d}.png")
                fig.savefig(fig_path)
                plt.close(fig)

    if cfg.checkpoint_dir is not None:
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        exp_state_path = os.path.join(cfg.checkpoint_dir, "experiment_state.pth")
        if os.path.exists(exp_state_path):
            logging.info("Loading experiment state from experiment_state.pth")
            state = torch.load(exp_state_path)
        else:
            logging.info("No experiment state found - starting from scratch")
            state = None
    else:
        state = None

    set_global_seed(cfg.seed)

    logging.info("Setting up model")

    model = get_model(OmegaConf.to_object(cfg))
    model = Proside(model, **cfg.get('model_kw', {}))
    model = ProstNFoundMeta(model, **cfg.get("metamodel", {}))

    model.to(cfg.device)
    if cfg.torch_compile:
        torch.compile(model)
    logging.info("Model setup complete")
    logging.info(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    logging.info(
        f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )
    if cfg.model_checkpoint:
        model_state = torch.load(cfg.model_checkpoint, map_location="cpu", weights_only=False)
        if "model" in model_state:
            model_state = model_state["model"]
        msg = model.load_state_dict(model_state, strict=False)
        logging.info(f"Loaded model from {cfg.model_checkpoint} with message `{msg}`.")
    if state is not None:
        model.load_state_dict(state["model"])

    # setup criterion
    if "pos_weight" not in cfg:
        cfg.pos_weight = 1.0
    criterion = build_loss(cfg)

    loaders = get_dataloaders_from_args(cfg.data)
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    tloaders = get_dataloaders_from_args(cfg.test_data)
    test_loader = tloaders["val"]
    # test_loader = loaders["test"]

    optimizer, lr_scheduler = setup_optimizer(cfg, model, train_loader)
    if state is not None:
        optimizer.load_state_dict(state["optimizer"])
        lr_scheduler.load_state_dict(state["lr_scheduler"])

    scaler = torch.cuda.amp.GradScaler()
    if state is not None:
        scaler.load_state_dict(state["gradient_scaler"])

    epoch = 0 if state is None else state["epoch"]
    logging.info(f"Starting at epoch {epoch}")
    best_score = 0 if state is None else state["best_score"]
    logging.info(f"Best score so far: {best_score}")
    if state is not None:
        rng_state = state["rng"]
        set_all_rng_states(rng_state)

    test_best_score = 0

    # model.return_dict = False

    def get_state():
        return {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_score": best_score,
            "gradient_scaler": scaler.state_dict(),
            "rng": get_all_rng_states(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "args": vars(cfg),
        }

    def save_checkpoint(name):
        state = get_state()
        if cfg.checkpoint_dir is not None:
            logging.info(f"Saving experiment snapshot to {cfg.checkpoint_dir}")
            torch.save(state, os.path.join(cfg.checkpoint_dir, name))
            if cfg.save_checkpoint_wandb:
                wandb.save(
                    os.path.join(cfg.checkpoint_dir, name),
                    base_path=cfg.checkpoint_dir,
                    policy="now",
                )
    def anttaseg(cfg):
        frozen_segnet = FrozenSegmentationModel(
            checkpoint_path='/datasets/exactvu_pca/checkpoint_store/Paul/medAI/MicroSegNet.pth',
            img_size=224,
            device='cuda'
            )
        return SupervisedTrainerWithSegTTT(model, frozen_segnet, cfg)

    if cfg.use_antta:
        ttt_seg_object = anttaseg(cfg)

    def tttmodels(cfg, model):
        if cfg.use_tent:
            tta_model, adapted_param_names, arch = setup_tent(
                model,
                lr=1e-3,
                steps=1,
                episodic=False,
                architecture="auto",
                layer_selection="auto",
                optimizer_cls=torch.optim.SGD,
                optimizer_kwargs={"momentum": 0.9},
            )
        elif cfg.use_eata:
            fishers = compute_fishers_from_dict_loader(
                model=model,
                fisher_loader=train_loader,
                device=cfg.device,
                num_samples=500,
            )
            tta_model, selected_names, arch = setup_eata(
                model=model,
                lr=1e-3,
                steps=1,
                episodic=False,
                fisher_alpha=2000.0,
                e_margin=math.log(2) * 0.4,
                d_margin=0.05,
                fishers=fishers,
                architecture="auto",
                layer_selection="auto",
                optimizer_cls=torch.optim.SGD,
                optimizer_kwargs={"momentum": 0.9},
            )
        elif cfg.use_sar:
            tta_model, selected_names, arch = setup_sar(
                model=model,
                lr=5e-4,
                steps=1,
                episodic=False,
                margin_e0=math.log(2) * 0.4,
                reset_constant_em=0.2,
                rho=0.05,
                sam_adaptive=False,
                base_optimizer_cls=torch.optim.SGD,
                base_optimizer_kwargs={"momentum": 0.9},
            )
        elif cfg.use_cotta:
            tta_model, adapted_param_names, arch = setup_cotta(
                model=model,
                lr=1e-3,
                steps=1,
                episodic=False,
                architecture="auto",
                layer_selection="auto",
                alpha_teacher=0.999,
                restoration_factor=0.01,
                augmentation_multiplicity=8,
                optimizer_cls=torch.optim.Adam,
                optimizer_kwargs={},
            )
        elif cfg.use_memo:
            tta_model, adapted_param_names, arch = setup_memo(
                model=model,
                lr = 2.5e-4,
                steps = 1,
                batch_size = 1,
                episodic = True,
                architecture = "auto",
                layer_selection = "auto",
                augmentation_type = "augmix",
                image_size = 512,
                mean=(0, 0, 0),
                std=(1, 1, 1),
            )
        elif cfg.use_petta:
            tta_model, selected_names, arch = setup_petta(
                model=model,
                lr = 1e-3,
                steps = 1,
                episodic = False,
                memory_size = 64,
                lambda_t = 1.0,
                lambda_u = 1.0,
                alpha_0 = 1e-3,
                lambda_0 = 10.0,
            )
        elif cfg.use_rotta:
            tta_model, selected_names, arch = setup_rotta(
                model = model,
                lr = 1e-3,
                steps = 1,
                episodic = False,
                memory_size = 64,
                update_frequency = 64,
                nu = 1e-3,
                lambda_t = 1.0,
                lambda_u = 1.0,
                bn_momentum = 0.05,
                mean=(0,0,0),
                std=(1,1,1),
            )
        elif cfg.use_roid:
            tta_model, selected_names, arch = setup_roid(
                model=model,
                lr = 2.5e-4,
                steps = 1,
                episodic = False,
                architecture = "auto",
                layer_selection = "auto",
                use_weighting = True,
                use_prior_correction = True,
                use_consistency= True,
                momentum_src = 0.99,
                momentum_probs = 0.9,
                temperature = 1.0 / 3.0,
                image_size = 512,
            )
        elif cfg.use_rmt:
            tta_model, selected_names, arch = setup_rmt(
                model=model,
                source_loader=train_loader,
                num_classes=2,
                lr = 1e-2,
                steps = 1,
                episodic = False,
                image_size = 512,
                lambda_ce_src = 1.0,
                lambda_ce_trg = 1.0,
                lambda_cont = 1.0,
                teacher_momentum = 0.999,
                temperature = 0.1,
                contrast_mode = "all",
                projection_dim = 128,
                warmup_samples = 0,
            )

        return tta_model



    # for epoch in range(epoch, cfg.epochs):
    #     if cfg.cutoff_epoch is not None and epoch > cfg.cutoff_epoch:
    #         break
    #     logging.info(f"Epoch {epoch}")

    #     save_checkpoint("experiment_state.pth")

    #     run_train_epoch(
    #         cfg, model, train_loader, criterion,
    #         optimizer, lr_scheduler, scaler, epoch,
    #         desc="train", log_fn=log_fn,
    #     )

    #     if cfg.run_val:
    #         model_state = copy.deepcopy(model.state_dict())  # save before TTA
    #         # Also save requires_grad flags and BN configs
    #         bn_configs = {}
    #         for name, module in model.named_modules():
    #             if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
    #                 bn_configs[name] = {
    #                     "track_running_stats": module.track_running_stats,
    #                     "running_mean": module.running_mean.clone() if module.running_mean is not None else None,
    #                     "running_var": module.running_var.clone() if module.running_var is not None else None,
    #                 }
        
    #         if cfg.use_antta:
    #             val_metrics, results_table = antta_eval_epoch(
    #                 cfg, model, val_loader, epoch, desc="val", log_fn=log_fn, 
    #                 ttt_seg_object=ttt_seg_object, model_state=model_state, bn_configs = bn_configs
    #             )
    #         else:              # re-wrap each epoch
    #             tta_model = tttmodels(cfg, model) 
    #             val_metrics, results_table = run_eval_epoch(
    #                 cfg, tta_model, val_loader, epoch, desc="val", log_fn=log_fn
    #             )
    #         model.load_state_dict(model_state)               # restore after TTA
    #         model.requires_grad_(True)
    #         # Restore BN configs that load_state_dict doesn't touch
    #         for name, module in model.named_modules():
    #             if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
    #                 if name in bn_configs:
    #                     module.track_running_stats = bn_configs[name]["track_running_stats"]
    #                     module.running_mean = bn_configs[name]["running_mean"]
    #                     module.running_var = bn_configs[name]["running_var"]

    #         if is_main_process() and cfg.output_dir is not None:
    #             table_path = os.path.join(
    #                 cfg.output_dir, f"val_results_epoch_{epoch:04d}.csv"
    #             )
    #             results_table.to_csv(table_path)

    #         if val_metrics is not None:
    #             tracked_metric = val_metrics[cfg.tracked_metric]
    #             new_record = tracked_metric > best_score
    #         else:
    #             new_record = None

    #         if new_record:
    #             best_score = tracked_metric
    #             logging.info(f"New best score: {best_score}")

    #         if cfg.run_test and (new_record or cfg.test_every_epoch):  # fixed precedence
    #             logging.info("Running test set")
    #             test_tta_model = tttmodels(cfg, model)       # fresh wrap for test
    #             metrics = run_eval_epoch(
    #                 cfg, test_tta_model, test_loader, epoch, desc="test", log_fn=log_fn
    #             )

    #         if new_record and cfg.save_best_weights:
    #             save_checkpoint("best.pth")


    # Final test on best val checkpoint
    if cfg.base_run_test:
        model_state = torch.load(
            # os.path.join(cfg.checkpoint_dir, 'best.pth'),
            os.path.join(cfg.model_checkpoint),
            map_location="cpu",
            weights_only=False,
        )
        if "model" in model_state:
            model_state = model_state["model"]
        msg = model.load_state_dict(model_state, strict=False)

        if cfg.use_antta:
            ttt_seg_object = anttaseg(cfg)
            model_state = copy.deepcopy(model.state_dict())  # save before TTA
            # Also save requires_grad flags and BN configs
            bn_configs = {}
            for name, module in model.named_modules():
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    bn_configs[name] = {
                        "track_running_stats": module.track_running_stats,
                        "running_mean": module.running_mean.clone() if module.running_mean is not None else None,
                        "running_var": module.running_var.clone() if module.running_var is not None else None,
                    }
            test_metrics, results_table = antta_eval_epoch(
                    cfg, model, test_loader, epoch, desc="test", log_fn=log_fn, 
                    ttt_seg_object=ttt_seg_object, model_state=model_state, bn_configs = bn_configs
            )
        else:
            tta_model = tttmodels(cfg, model)                    # fresh wrap on best weights
            test_metrics, results_table = run_eval_epoch(
                cfg, tta_model, test_loader, epoch, desc="test", log_fn=log_fn
            )

        if is_main_process() and cfg.output_dir is not None:
            table_path = os.path.join(
                cfg.output_dir, f"test_results_epoch_{epoch:04d}.csv"
            )
            results_table.to_csv(table_path)

        if test_metrics is not None:
            tracked_metric = test_metrics[cfg.test_tracked_metric]
            new_record = tracked_metric > test_best_score
        else:
            new_record = None

        if new_record:
            test_best_score = tracked_metric
            logging.info(f"New best score: {test_best_score}")

        if new_record and cfg.save_best_weights:
            save_checkpoint("best_test.pth")

    logging.info("Finished training")

def run_train_epoch(
    args,
    model,
    loader,
    criterion,
    optimizer,
    scheduler,
    scaler,
    epoch,
    desc="Train",
    log_fn=None,
):
    # setup epoch
    model.train()
    evaluator = Evaluator(**args.evaluator)

    for train_iter, data in enumerate(tqdm(loader, desc=desc)):

        if args.debug and train_iter > 10:
            break

        # if train_iter > 200:
        #     break

        # run the model
        with torch.cuda.amp.autocast(enabled=args.use_amp):

            data = model(data)  # heatmap

            if torch.any(torch.isnan(data["cancer_logits"])):
                logging.warning("NaNs in heatmap logits")

            # loss calculation
            loss = criterion(data)

        loss = loss / args.accumulate_grad_steps
        # backward pass
        if args.use_amp:
            logging.debug("Backward pass")
            scaler.scale(loss).backward()
        else:
            logging.debug("Backward pass")
            loss.backward()

        # gradient accumulation and optimizer step
        if args.debug:
            for param in optimizer.param_groups[1]["params"]:
                break
            logging.debug(param.data.view(-1)[0])

        if (train_iter + 1) % args.accumulate_grad_steps == 0:
            logging.debug("Optimizer step")
            if args.use_amp:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else:
                optimizer.step()
                optimizer.zero_grad()

            if args.debug:
                for param in optimizer.param_groups[1]["params"]:
                    break
                logging.debug(param.data.view(-1)[0])

        scheduler.step()

        # accumulate outputs
        step_metrics = {f"train/{k}": v for k, v in evaluator(data).items()}

        # log metrics
        step_metrics.update({"train_loss": loss.item() / args.accumulate_grad_steps})
        encoder_lr = optimizer.param_groups[0]["lr"]
        main_lr = optimizer.param_groups[1]["lr"]
        cnn_lr = optimizer.param_groups[2]["lr"]
        step_metrics["encoder_lr"] = encoder_lr
        step_metrics["main_lr"] = main_lr
        step_metrics["cnn_lr"] = cnn_lr

        if log_fn is not None:
            log_fn(step_metrics)

    # compute and log metrics
    metrics = evaluator.aggregate_metrics()
    desc = "train"
    metrics = {f"{desc}/{k}": v for k, v in metrics.items()}
    metrics["epoch"] = epoch
    if log_fn is not None:
        log_fn(metrics)


def run_eval_epoch(args, model, loader, epoch, desc="eval", log_fn=None, ttt_seg_object=None):
    model.eval()
    batch_size = args.data.batch_size
    evaluator = Evaluator(**args.evaluator)
    ttt_times_per_image = []

    for val_iter, data in enumerate(tqdm(loader, desc=desc)):
        # Standard validation (no TTT)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.cuda.amp.autocast(enabled=args.use_amp):
            with torch.no_grad():
                data = model(data)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        ttt_times_per_image.append((t1 - t0) / batch_size)

        # accumulate outputs
        step_metrics = {f"{desc}/{k}": v for k, v in evaluator(data).items()}
        if step_metrics and log_fn is not None:
            log_fn(step_metrics)

    if args.use_ttt:
        # ttt_seg_object.reset_objects()
        mean_ttt_time = np.mean(ttt_times_per_image)
        std_ttt_time = np.std(ttt_times_per_image)
        wandb.log({
            "ttt/mean_ttt_time_per_image_s": mean_ttt_time,
            "ttt/std_ttt_time_per_image_s": std_ttt_time,
            "ttt/epoch": epoch,
        })

    metrics = evaluator.aggregate_metrics()
    results_table = evaluator.results_table
    
    bootstrap_metrics = compute_bootstrap_metrics(results_table, desc)
    metrics.update(bootstrap_metrics)

    metrics = {f"{desc}/{k}": v for k, v in metrics.items()}
    metrics["epoch"] = epoch
    if log_fn is not None:
        log_fn(metrics)

    return metrics, results_table

def antta_eval_epoch(args, model, loader, epoch, desc="eval", log_fn=None, ttt_seg_object=None,  
    model_state=None, bn_configs = None,
    ):
    model.eval()

    evaluator = Evaluator(**args.evaluator)
    dice_scores = []
    for val_iter, data in enumerate(tqdm(loader, desc=desc)):
        if args.use_ttt:
            # Clone and adapt model via segmentation TTT
            adapted_model = copy.deepcopy(model)
            adapted_model = ttt_seg_object.reconfigure_model(adapted_model)
            adapted_model, dice_score = ttt_seg_object.apply_segmentation_ttt(adapted_model, data, val_iter)
            dice_scores.append(dice_score)
            
            # Predict with adapted model
            adapted_model.eval()
            with torch.cuda.amp.autocast(enabled=args.use_amp):
                with torch.no_grad():
                    data = adapted_model(data)
            
            del adapted_model
            torch.cuda.empty_cache()
            
            # Restore original model
            # if model_state is not None:
            model.load_state_dict(model_state)
            # if bn_configs is not None:
            for name, module in model.named_modules():
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    if name in bn_configs:
                        module.track_running_stats = bn_configs[name]["track_running_stats"]
                        module.running_mean = bn_configs[name]["running_mean"]
                        module.running_var = bn_configs[name]["running_var"]

        else:
            # Standard validation (no TTT)
            with torch.cuda.amp.autocast(enabled=args.use_amp):
                with torch.no_grad():
                    data = model(data)

        # accumulate outputs
        step_metrics = {f"{desc}/{k}": v for k, v in evaluator(data).items()}
        if step_metrics and log_fn is not None:
            log_fn(step_metrics)
    
    if args.use_ttt:
        wandb.log({
            "ttt/epoch_dice": np.mean(dice_scores),
            "ttt/epoch": epoch
        })

    metrics = evaluator.aggregate_metrics()
    results_table = evaluator.results_table
    metrics = {f"{desc}/{k}": v for k, v in metrics.items()}
    metrics["epoch"] = epoch
    if log_fn is not None:
        log_fn(metrics)

    return metrics, results_table



def setup_optimizer(args, model, train_loader):
    from torch.optim import AdamW

    (
        encoder_parameters,
        warmup_parameters,
        cnn_parameters,
    ) = model.get_params_groups()

    total_epochs = args.epochs
    encoder_frozen_epochs = args.warmup_epochs
    warmup_epochs = 5
    niter_per_ep = len(train_loader)
    warmup_lr_factor = args.warmup_lr / args.lr
    params = [
        {"params": encoder_parameters, "lr": args.encoder_lr},
        {"params": warmup_parameters, "lr": args.lr},
        {"params": cnn_parameters, "lr": args.cnn_lr},
    ]

    def compute_lr_multiplier(iter, is_encoder_or_cnn=True):
        schedule = args.get("scheduler", "cosine")
        if schedule == "constant":
            return 1

        if iter < encoder_frozen_epochs * niter_per_ep:
            if is_encoder_or_cnn:
                return 0
            else:
                if iter < warmup_epochs * niter_per_ep:
                    return (iter * warmup_lr_factor) / (warmup_epochs * niter_per_ep)
                else:
                    cur_iter_in_frozen_phase = iter - warmup_epochs * niter_per_ep
                    total_iter_in_frozen_phase = (
                        encoder_frozen_epochs - warmup_epochs
                    ) * niter_per_ep
                    return (
                        0.5
                        * (
                            1
                            + np.cos(
                                np.pi
                                * cur_iter_in_frozen_phase
                                / (total_iter_in_frozen_phase)
                            )
                        )
                        * warmup_lr_factor
                    )
        else:
            iter -= encoder_frozen_epochs * niter_per_ep
            if iter < warmup_epochs * niter_per_ep:
                return iter / (warmup_epochs * niter_per_ep)
            else:
                cur_iter = iter - warmup_epochs * niter_per_ep
                total_iter = (
                    total_epochs - warmup_epochs - encoder_frozen_epochs
                ) * niter_per_ep
                return 0.5 * (1 + np.cos(np.pi * cur_iter / total_iter))

    optimizer = AdamW(params, lr=args.lr, weight_decay=args.wd)
    from torch.optim.lr_scheduler import LambdaLR

    lr_scheduler = LambdaLR(
        optimizer,
        [
            lambda iter: compute_lr_multiplier(iter, is_encoder_or_cnn=True),
            lambda iter: compute_lr_multiplier(iter, is_encoder_or_cnn=False),
            lambda iter: compute_lr_multiplier(iter, is_encoder_or_cnn=True),
        ],
    )

    return optimizer, lr_scheduler


class ProstNFoundMeta(nn.Module):
    """Wraps a model to perform forward pass with ProstNFound style training

    Args:
        model: The model to wrap.
        mask_output_key: The key to use for the mask output (if the model outputs a dictionary of tensors)
    """

    def __init__(self, model: nn.Module, mask_output_key=None):
        super().__init__()
        self.model = model
        self.mask_output_key = mask_output_key

        if isinstance(self.model, ProstNFound):
            logging.info(f"Model ProstNFound with prompts {self.model.prompts}")

        self.register_buffer("temperature", torch.tensor([1.0]))
        self.register_buffer("bias", torch.tensor([0.0]))

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, data, include_postprocessed_heatmaps=False):
        # extracting relevant data from the batch
        bmode = data["bmode"].to(self.device)
        needle_mask = data["needle_mask"].to(self.device)
        prostate_mask = data["prostate_mask"].to(self.device)
        if "rf" in data:
            rf = data["rf"].to(self.device)
        else:
            rf = None

        B = len(bmode)

        # Wrapped forward pass
        if isinstance(self.model, ProstNFound):
            prompts = {}
            for prompt_name in self.model.prompts:
                prompts[prompt_name] = data[prompt_name].to(
                    device=self.device, dtype=bmode.dtype
                )
                if prompts[prompt_name].ndim == 1:
                    prompts[prompt_name] = prompts[prompt_name][:, None]

            outputs = self.model(
                bmode, rf, prostate_mask, needle_mask, output_mode="all", **prompts
            )
            cancer_logits = outputs["mask_logits"]
            image_level_classification_outputs = outputs["cls_outputs"]
            data["image_level_classification_outputs"] = (
                image_level_classification_outputs
            )

        elif isinstance(self.model, Proside):
            prompts = {}
            for prompt_name in self.model.discrete_prompts:
                prompts[prompt_name] = data[prompt_name].to(
                    device=self.device, dtype=bmode.dtype
                )
                if prompts[prompt_name].ndim == 1:
                    prompts[prompt_name] = prompts[prompt_name][:, None]

            outputs = self.model(
                bmode, output_mode="all", **prompts
            )
            cancer_logits = outputs["mask_logits"]
            image_level_classification_outputs = outputs["cls_outputs"]
            data["image_level_classification_outputs"] = (
                image_level_classification_outputs
            )

        else:
            model_outputs = self.model(bmode)
            if isinstance(model_outputs, dict):
                cancer_logits = model_outputs[self.mask_output_key]
            else:
                cancer_logits = self.model(bmode)

        cancer_logits = (
            cancer_logits / self.temperature[None, None, None, :]
            + self.bias[None, None, None, :]
        )
        data["cancer_logits"] = cancer_logits

        # compute predictions
        masks = (prostate_mask > 0.5) & (needle_mask > 0.5)
        predictions, batch_idx = MaskedPredictionModule()(cancer_logits, masks)
        mean_predictions_in_needle = []
        for j in range(B):
            mean_predictions_in_needle.append(
                predictions[batch_idx == j].sigmoid().mean()
            )
        mean_predictions_in_needle = torch.stack(mean_predictions_in_needle)
        data["average_needle_heatmap_value"] = mean_predictions_in_needle

        prostate_masks = prostate_mask > 0.5
        predictions, batch_idx = MaskedPredictionModule()(cancer_logits, prostate_masks)
        mean_predictions_in_prostate = []
        for j in range(B):
            mean_predictions_in_prostate.append(
                predictions[batch_idx == j].sigmoid().mean()
            )
        mean_predictions_in_prostate = torch.stack(mean_predictions_in_prostate)
        data["average_prostate_heatmap_value"] = mean_predictions_in_prostate

        if include_postprocessed_heatmaps:
            cancer_logits = data["cancer_logits"]
            heatmap = cancer_logits[0, 0].detach().sigmoid().cpu().numpy()
            heatmap = (heatmap * 255).astype(np.uint8)
            # blur and upsample
            import cv2

            blurred = cv2.GaussianBlur(heatmap, (5, 5), sigmaX=1.5)
            upsampled = cv2.resize(blurred, (256, 256), interpolation=cv2.INTER_LINEAR)
            heatmap = upsampled
            data["cancer_probs"] = (torch.tensor(heatmap) / 255.0)[None, None, ...]

        return data

    def get_params_groups(self):
        if isinstance(self.model, SETR):
            encoder_parameters = []
            warmup_parameters = []
            cnn_parameters = []
            for name, param in self.model.named_parameters():
                if "head" in name:
                    warmup_parameters.append(param)
                else:
                    encoder_parameters.append(param)
            return encoder_parameters, warmup_parameters, cnn_parameters

        elif isinstance(self.model, ProstNFound):
            return self.model.get_params_groups()

        elif hasattr(self.model, "image_encoder"):
            encoder_parameters = []
            warmup_parameters = []
            cnn_parameters = []
            for name, param in self.model.named_parameters():
                if "image_encoder" in name:
                    encoder_parameters.append(param)
                else:
                    warmup_parameters.append(param)
            return encoder_parameters, warmup_parameters, cnn_parameters

        elif hasattr(self.model, "get_params_groups"):
            return self.model.get_params_groups()

        else:
            from itertools import chain

            encoder_parameters = []
            warmup_parameters = self.model.parameters()
            cnn_parameters = []
            return encoder_parameters, warmup_parameters, cnn_parameters


def load_config(config_path, options):
    cfg = OmegaConf.load(config_path)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(options))

    # deal with issues from old schemas
    if cfg.tracked_metric == "val/core_auc_high_involvement":
        warn("`val/core_auc_high_involvement` is deprecated - use `val/auc` instead")
        cfg.tracked_metric = "val/auc"

    return cfg


if __name__ == "__main__":
    # main()
    p = ArgumentParser(description="Train ProstNFound model")
    p.add_argument(
        "--config", "-c", help="Path to config file (located in cfg/train/...)"
    )
    p.add_argument("options", nargs=argparse.REMAINDER)
    args = p.parse_args()
    cfg = load_config(args.config, args.options)

    main(cfg)