"training flows from template generator"
import os
import sys
import logging

from datetime import datetime
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pandas as pd
import joblib
import torch
import hydra
from torch import optim
from torch.utils.data import DataLoader
from nflows.transforms.base import InputOutsideDomain
from sklearn.model_selection import train_test_split

## internal imports
from otcalib.otcalib.torch.torch_utils as utils
from otcalib.otcalib.utils.transformations as trans
from src.datamodule import get_samples
from src.get_flow import stacked_norm_flow

# TODO add to package
from tools.create_log_file import CreateLog
from tools.misc import save_fig, save_yaml
from tools.schedulers import get_scheduler
from tools.discriminator import DenseNet
from tools.visualization import cflow_plotting as visual

def train_template_generator(
    config,
    data_name:str,
    device="cuda", 
    **kwargs,
):
    """train flow for template generator

    Parameters
    ----------
    config : dict
        config dict
    label_name : str
        ftag_sig or ftag_bkg
    save_path : str, optional
        where to save it, by default None
    old_model : str, optional
        if old model should be load, by default None

    Returns
    -------
    str
        output the path to the flow
    """
    save_path = config["path"]["save_path"]
    # if old_model is not None:
    #     outdir = old_model
    # else:
    outdir = (
        save_path
        + f"/{data_name}"
        + f"_{datetime.today().strftime('%m_%d_%Y_%H_%M_%S_%f')}"
    )

    #### DATA SETUP ####
    data, scaler, weights, _ = get_samples(
        config['data'], flow_cfg=config['model']['flow'],
        data_name=data_name,
        trans_lst= kwargs.get('trans_lst')
        )
    
    # splitting data and creating sampler for weights
    # data = scaler.transform(np.float32(data))
    weights = np.ravel(weights)
    data = np.c_[data, weights]
    if config['data'].get("maxevents", None) is not None:
        data = data[:config['data']["maxevents"]]

    # apply weights to test sample
    x_train, x_test = train_test_split(data, test_size=config['data'].get("valid_size",0.2))

    train_dataloader = DataLoader(x_train, **config['train']["dataloader_kwargs"])
    test_dataloader = DataLoader(x_test, **config['train']["dataloader_kwargs"])


    logging.info("Training size: %s", x_train.shape)
    logging.info("Test size: %s", x_test.shape)
    #### FLOW SETUP ####
    cvx_dims = len(config['data']['condnames'])
    noncvx_dims = len(config['data']['transportnames'])
    transport_idx = [noncvx_dims, cvx_dims+noncvx_dims]

    # init flow
    flow = hydra.utils.instantiate(config['model']['flow']).to(device)

    trainable_params = np.sum([i.numel() for i in flow.parameters()
                               if i.requires_grad])
    logging.info(f"Trainable parameters: {trainable_params}")

    # init training
    flow_train_cfg = config['train']["flow"]
    optimizer = optim.Adam(flow.parameters(), lr=flow_train_cfg["lr"])
    sch_attr = flow_train_cfg["sch_config"]
    sch_attr["T_max"] = flow_train_cfg["n_epochs"] * len(train_dataloader)
    scheduler = get_scheduler("singlecosine", optimizer, attr=sch_attr)

    log_file = CreateLog(outdir, ["loss", "lr", "mean_clip"],)
    epoch_start_nr = len(log_file.log["loss"])

    # create folders
    for i in [
        "",
        "models",
        "figures",
        "figures/latent",
        "figures/1d_marginals",
        "figures/dl1r",
    ]:
        os.makedirs(f"{outdir}/{i}", exist_ok=True)
    save_yaml(config, outdir + "/config.yaml", hydra=True)

    if scaler is not None:
        joblib.dump(scaler, f"{outdir}/scaler.json")

    plotting = visual.Plotting(
        flow,
        test_sample = torch.tensor(x_test).float()[:,noncvx_dims:noncvx_dims+cvx_dims],
        test_sample_conds = torch.tensor(x_test).float()[:,:noncvx_dims],
        weights=x_test[:, noncvx_dims+cvx_dims],
        device=device,
        n_times_sample=1,
        percentile_lst=[0, 100],
    )

    target_names = config['data']['transportnames']

    plotting.ylim = [0.75, 1.25]
    pbar = tqdm(range(epoch_start_nr, flow_train_cfg["n_epochs"]))

    total_loss = log_file.log["loss"].copy()
    valid_loss = []

    ##### train flow #####
    for i in pbar:

        loss_track = []
        clip_lst = []
        for data_iter in train_dataloader:
            context = data_iter[:, :noncvx_dims].float().to(device)
            target_dist = data_iter[:, transport_idx[0]:transport_idx[1]].float().to(device)
            weights = data_iter[:, -1].float().to(device)
            optimizer.zero_grad()
            try: 
                loss = torch.mean(
                    -weights*flow.log_prob(inputs=target_dist, context=context)
                    )
            except InputOutsideDomain:
                print(f"InputOutsideDomain erorr at epoch {i}")
                print(f"Inputs min: {target_dist.min(0)}")
                print(f"Inputs max: {target_dist.max(0)}")
                continue
            loss_track.append(loss.cpu().detach().numpy())
            loss.backward()
            clip_values = torch.nn.utils.clip_grad_norm_(flow.parameters(), 5)
            optimizer.step()
            scheduler.step()
            clip_lst.append(clip_values.cpu().numpy())

        epoch_loss = np.float64(np.mean(loss_track))
        total_loss.append(epoch_loss)
        _valid_loss=[]
        for test_d in test_dataloader:
            context = test_d[:, :noncvx_dims].float().to(device)
            target_dist = test_d[:, transport_idx[0]:transport_idx[1]].float().to(device)
            weights = test_d[:, -1].float().to(device)
            _valid_loss.extend(
                (-weights*flow.log_prob(
                    inputs=target_dist,
                    context=context,
                )).cpu().detach().numpy()
                )
        valid_loss.append(np.mean(_valid_loss))
            

        if (i == 0) or (np.min(valid_loss[:-1]) > valid_loss[-1]):
            best_epoch = i
            try:
                plotting.update_sample(flow)
                plotting.marginals()
                fig, _, _ = plotting.plot_marginals(xlabel=target_names, y_scale="log")
                # sys.exit()
                save_fig(
                    fig,
                    f"{outdir}/figures/1d_marginals/epoch_nr_{best_epoch}.png",
                    close_fig=True,
                )
            except InputOutsideDomain:
                print(f"InputOutsideDomain erorr at epoch {i}")
                print(f"Inputs min: {target_dist.min(0)}")
                print(f"Inputs max: {target_dist.max(0)}")
            torch.save(
                {
                    "flow": flow.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                },
                f"{outdir}/models/model_setup_{best_epoch}.pt",
            )

        pbar.set_postfix({"Flow loss": f"{str(round(total_loss[-1],4))}"})

        fig = plt.figure()
        plt.plot(total_loss, label="Flow loss")
        plt.plot(valid_loss, label="valid loss")
        plt.xlabel("Epoch number")
        plt.ylabel("Loss")
        if np.min(total_loss) > 0:
            plt.yscale("log")
        save_fig(fig, f"{outdir}/figures/flow_loss.png", close_fig=True)

        log = dict(
            zip(
                log_file.keys(),
                [epoch_loss, optimizer.param_groups[0]["lr"], np.mean(clip_lst)],
            )
        )
        log_file.update_log(log)
        log_file.save()

    # load best flow
    model_state = torch.load(
        f"{outdir}/models/model_setup_{best_epoch}.pt", map_location=device
    )
    flow.load_state_dict(model_state["flow"])

    ##### evaluate using classifier #####
    dense_network = kwargs.get("dense_network", {})
    dense_network["input_dim"] = cvx_dims+noncvx_dims
    dense_network["sigmoid"] = False
    dense_network["using_weights"] = cvx_dims+noncvx_dims < data.shape[-1]
    classifier = DenseNet(**dense_network)

    # create data samples
    flow_gen_sample = [torch.concat([i, 
        flow.sample(1, i.float().to(device))[:, 0, :None].cpu().detach()], 1)
        for i in tqdm(
            torch.tensor(data[:, :noncvx_dims]).chunk(
                (len(data)//512)+1
            )
        )
    ]

    flow_gen_sample = torch.concat(flow_gen_sample,0).numpy()

    flow_gen_sample = scaler.inverse_transform(flow_gen_sample)

    # add weight to second to last columns
    if cvx_dims+noncvx_dims < data.shape[-1]:
        clf_data = np.c_[scaler.inverse_transform(data[:, :cvx_dims+noncvx_dims]), data[:, -1], np.ones((len(data), 1))]
        flow_gen_sample = np.c_[flow_gen_sample, data[:, -1], np.zeros((len(flow_gen_sample), 1))]
    else:
        clf_data = np.c_[scaler.inverse_transform(data), np.ones((len(data), 1))]
        flow_gen_sample = np.c_[flow_gen_sample, np.zeros((len(flow_gen_sample), 1))]

    X_train, X_test = train_test_split(
        torch.tensor(np.r_[clf_data, flow_gen_sample]).float(),
        test_size=0.25)
    
    # cap the clf training size
    if len(X_train)> 5_000_000:
        X_train = X_train[:5_000_000]
        
    if len(X_test)> 5_000_000:
        X_test = X_test[:5_000_000]

    # create dataloaders
    train_dataloader_clf = DataLoader(X_train.detach(), **config['train']['dataloader_kwargs'])

    valid_dataloader_clf = DataLoader(X_test.detach(), **config['train']['dataloader_kwargs'])

    classifier.run_training(
        train_dataloader_clf, valid_loader=valid_dataloader_clf, n_epochs=config['train']["dense"].get("epochs", 50)
    )

    # save clf
    save_yaml(classifier.loss_data, f"{outdir}/clf_logging.yml")

    # plot auc
    fig = plt.figure()
    plt.plot(classifier.loss_data["valid_auc"], label="Valid AUC")
    plt.legend()

    save_fig(
        fig,
        f"{outdir}/figures/valid_auc.png",
        close_fig=True,
    )

    # eval high stat TODO do not work with weights
    plotting.test_sample=flow_gen_sample[:, transport_idx[0]:transport_idx[1]]
    plotting.sample = clf_data[:, transport_idx[0]:transport_idx[1]]
    if cvx_dims+noncvx_dims < clf_data.shape[-1]:
        plotting.weights = clf_data[:, -1]
        plotting.sample_weights = clf_data[:, -1]

    plotting.marginals()
    fig, _, _ = plotting.plot_marginals(xlabel=target_names, y_scale="log")

    save_fig(
        fig,
        f"{outdir}/figures/1d_marginals/epoch_nr_high_stat.png",
        close_fig=True,
    )
    if "ftag" in config['data']:
        fig = plotting.plot_dl1r(target_names)
        save_fig(
            fig,
            f"{outdir}/figures/dl1r/dl1r_epoch_nr_high_stat.png",
            close_fig=True,
        )
    return outdir, np.max(classifier.loss_data["valid_auc"])
