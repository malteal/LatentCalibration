"""load flow and sample correct conditional template.
Determine the signal to background probability"""
import pyrootutils

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)

import os
import sys
import logging
from glob import glob
from copy import deepcopy

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
import logging
from torch.utils.data import DataLoader
import hydra
from omegaconf import OmegaConf, DictConfig
from sklearn.model_selection import train_test_split
from otcalib.otcalib.utils import transformations as trans 

# internal imports
from src.datamodule import get_samples, TemplateGenerator

# ot-framework imports
from otcalib.otcalib.utils import plotutils
from otcalib.otcalib.torch.loader as loader
from otcalib.otcalib.torch.torch_utils as ot_utils
from otcalib.otcalib.utils import transformations as trans 
from otcalib.otcalib.utils.misc as misc 


# separate imports
from tools.discriminator import DenseNet
import tools.flows as flows
# from tools.visualization.atlas_utils import get_atlas_internal_str, get_atlas_legend, ATLAS_setup
from tools.visualization import general_plotting as plot

def evaluate_density(classifer: torch.nn.Module, valid_data: torch.Tensor, valid_labels: torch.Tensor, save_path: str, col_names: list=None,
                     **kwargs):
    """
    Evaluates and plots the density ratio of a multi-classifier.

    Parameters:
    - classifer (torch.nn.Module): The classifier model to evaluate.
    - valid_data (torch.Tensor): The validation data to evaluate the classifier on.
    - valid_labels (torch.Tensor): The labels corresponding to the validation data.
    - save_path (str): The path where the plots will be saved.
    - col_names (list, optional): The column names for the validation data. Defaults to None.
    - **kwargs: Additional keyword arguments.

    The plots are saved in the specified save_path under the "density_ratio" directory.
    """

    if len(classifer.loss_data)>0:
        for i in classifer.loss_data:
            fig = plt.figure()
            plt.plot(np.ravel(classifer.loss_data[i]), label=i)
            plt.xlabel("Epochs")
            plt.ylabel(i.split("_")[-1])
            plt.legend()
            misc.save_fig(fig, f"{save_path}/density_ratio/{i}.png")

    output = []
    n_chunks = valid_data.chunk((len(valid_data)//128)+1)
    for i in tqdm(n_chunks, total=len(n_chunks), leave=False):
        output.append(torch.sigmoid(classifer.forward(i.to(classifer.device))[-1]).cpu().detach())

    output = torch.cat(output, 0)
    output = np.ravel(output.cpu().detach().numpy())
    valid_data = valid_data.cpu().detach().numpy()
    valid_labels = valid_labels.cpu().detach().numpy()

    fig = plt.figure()
    plt.hist(output, bins=100)
    misc.save_fig(fig, f"{save_path}/density_ratio/hist_proba.png")

    print(f"Accuracy: {np.mean((valid_labels == 1)*1.0)}")

    if col_names is None:
        col_names = [f"toy_{i}" for i in range(valid_data.shape[-1])]

    for i in range(valid_data.shape[-1]):

        idx_sort = np.argsort(valid_data[:, i])
        
        data = np.c_[valid_data[:, i], valid_labels]

        plot_density_ratio(data=data, sorted_conds=valid_data[idx_sort, i],
                           pred_prob=output[idx_sort], save_path=save_path, save_bool=True,
                           xlabel=col_names[i])
    

def plot_density_ratio(data, sorted_conds, pred_prob, save_path, save_bool=False,
                   plot_kwargs=None, xlabel:str=''):
    """
    data: np.array with [log(pt), label]
    
    sorted_conds: np.array of log(pt)
    
    pred_prob: np.array of predicted probability from clf
    
    
    """
    
    # get label
    pred_label = pred_prob >= np.random.uniform(size=len(pred_prob))
    
    
    # define plotting style
    if plot_kwargs is None:
        plot_kwargs = {"bins":100, "stacked":True, "density":False, "histtype":"step"}
        plot_kwargs["range"]=np.percentile(data[:, 0], (0.01, 99.9))
    
    # get the density ratio from histograms
    fig = plt.figure()
    counts, bins, _ = plt.hist(
        [
            data[:, 0][data[:, -1] == 0],
            data[:, 0][data[:, -1] == 1],
        ],
        label=["Background", "Signal"],
        **plot_kwargs
    )
    plt.hist(
        [
            sorted_conds[~pred_label].flatten(),
            sorted_conds[pred_label].flatten(),
        ],
        label=["Predicted background", "Predicted signal"],
        bins=bins,
        stacked=True,
        density=True,
        histtype="step",
        ls="dashed",
    )
    # plt.legend(title=get_atlas_legend())
    plt.xlabel(xlabel)
    plt.ylabel("Normalised entries")
    plt.tight_layout()
    if save_bool:
        misc.save_fig(fig, f"{save_path}/density_ratio/hist_{xlabel}.pdf")

    # plot continuous density ratio
    sig_counts = (counts[1] - counts[0]) / counts[1]
    yerr = np.sqrt(sig_counts*(1-sig_counts)/counts[1])
    xerr = (bins[1:] - bins[:-1]) / 2
    x_bins = (bins[1:] + bins[:-1]) / 2
    fig = plt.figure()
    plt.errorbar(
        x_bins, sig_counts, xerr=xerr, yerr=yerr, label=r"Signal probability", color="blue"
    )
    plt.plot(sorted_conds, pred_prob, label=r"NN signal probability", color="red", zorder=10)
    # plt.legend(title=get_atlas_legend(), frameon=False, loc="best")
    plt.xlim(plot_kwargs.get("range", None))
    plt.xlabel(xlabel)
    plt.ylabel("Probability")
    # plt.xscale("log")
    plt.tight_layout()
    if save_bool:
        misc.save_fig(fig, f"{save_path}/density_ratio/f_sig_{xlabel}.pdf")


def train_density_estimator(
    config: DictConfig,
    data: np.ndarray=None,
    weights: np.ndarray=None,
) -> None:
    """train a nn to estimate the sig/bkg ratio of the conds

    Parameters
    ----------
    config : DictConfig
        config
    sig_flow_path : str
        path to the signal flow
    bkg_flow_path : str, optional
        path to the bkg flow, by default None
    dataloader_style : dict, optional
        dataloader behavoir, by default None
    """
    # load signal flow args
    save_path = config['path']['save_path']

    if data is None:
        config['data'].pop('transportnames', None)
        #### DATA SETUP ####
        data, scaler, weights, _ = get_samples(config['data'])
    
        # splitting data and creating sampler for weights

        data = np.c_[data, np.ravel(weights)]
        
        # Swap weight and labels
        data[:, [-2, -1]] = data[:, [-1, -2]]

        if config['data'].get("maxevents", None) is not None:
            data = data[:config['data']["maxevents"]]

    # apply weights to test sample
    x_train, x_test = train_test_split(np.float32(data), test_size=config['data'].get("valid_size",0.2))
    
    logging.info(f"Train size: {x_train.shape}")
    logging.info(f"Test size: {x_test.shape}")

    train_dataloader = DataLoader(x_train, **config['train']['dataloader_kwargs'])
    valid_dataloader = DataLoader(x_test, **config['train']['dataloader_kwargs'])

    # clf args
    dense_cfg = config["model"]["dense"]

    if 'input_dim' not in dense_cfg:
        dense_cfg["input_dim"] = len(config["data"]["condnames"])

    ### run clf ###
    dense_net = DenseNet(**config["model"]["dense"])

    # train clf
    misc.save_yaml(dense_cfg, f"{save_path}/model_config.yaml", hydra=True)
    dense_net.run_training(train_dataloader, valid_dataloader,
                            n_epochs=config["train"]["dense"]["epochs"],
                        standard_lr_scheduler=True,
                        load_best=True)
    dense_net.save(output_path=f"{save_path}/models/model_and_logging.pt")

    if x_test.shape[-1]>len(config['data']['condnames'])+1:
        logging.info('Appling weight to test sample using |weight|*sum(|weight|)/sum(|weight|)')
        
        weights = np.ravel(x_test[:,-2])
        
        idx = misc.generate_idx_given_probs(np.abs(weights)*np.abs(weights).sum()/weights.sum(), len(weights))
        
        x_test = x_test[idx]
        

    evaluate_density(classifer=dense_net,
                 valid_data=torch.tensor(x_test[:,:dense_cfg["input_dim"]]), 
                 valid_labels=torch.tensor(x_test[:,-1]),
                 save_path=save_path,
                 col_names=config["data"]["condnames"]
                 )

def evaluate_template_performance(
    data_path,
    generator_path,
    size=16_000_000,
    bkg_bool=True,
    device="cuda",
    dense_network={},
    signal_label:list=None,
):
    # raise NotImplementedError("This function is not implemented -  mc_sample[:, -1] == 5 should be changed")

    "Evaluate the combined template generator vs the training sample"
    # if not isinstance(data_path, list):
    #     new_signal_model = None
    if signal_label is None:
        signal_label = [5]
    sig_path = misc.load_yaml(f"{generator_path}/data_setting.yml")["sig_path"]
    # data_args = flows.load_path_info(glob(f"{generator_path}/*sig*")[0], yaml_bool=False)[0]
    data_args = flows.load_path_info(sig_path, yaml_bool=False)[0]

    data_args.update({
        "maxevents": size,
        "probnames": ["jet_DL1r_pb", "jet_DL1r_pc", "jet_DL1r_pu"],
        "condnames": ["jet_pt"],
        "train_size": 0,
        "valid_size": size,
        "bkg": bkg_bool,
        "tnames": data_path,
        "additional_columns": ["jet_truthflav"],
        "use_weights": ["abs_event_xsec"],
    })
    data_args = OmegaConf.to_object(data_args)
    mc_sample, _, weights, _ = get_data(add_additional_columns=False,**data_args)  # sample are shuffled
    mc_sample[:, -1] = np.isin(mc_sample[:, -1], signal_label)  

    mc_sample = mc_sample[
        ~np.any(np.abs(mc_sample) == np.inf, 1)
    ]

    print(f"MC sample/weights size {len(mc_sample)}/{len(weights)}")

    weight_index = np.random.choice(
        len(mc_sample),
        len(mc_sample),
        p=np.ravel(weights / np.sum(weights)),
    )

    weighted_sample = mc_sample[weight_index]

    flow_gen = SampleFromPT(
        torch.tensor(weighted_sample[:, :1]).float(),
        generator_path,
        generate_bkg=bkg_bool,
        device=device,
        batch_size=1024,
        convex_dim=3,
        chunk_size=25_000,
    )
    dense_network["input_dim"] = weighted_sample.shape[1]
    dense_network["sigmoid"] = True
    classifier = DenseNet(**dense_network)

    clf_data = np.r_[
        np.c_[weighted_sample, np.ones((len(weighted_sample),))],
        np.c_[flow_gen.data.detach().numpy(), np.zeros((len(flow_gen.data),))],
    ]

    X_train, X_test = train_test_split(  # pylint: disable=C0103
        clf_data, test_size=0.25, shuffle=True
    )

    dataloader_args = {"shuffle": True, "drop_last": True, "batch_size": 1024}
    train_dataloader_clf = DataLoader(torch.tensor(X_train).float(), **dataloader_args)
    valid_dataloader_clf = DataLoader(torch.tensor(X_test).float(), **dataloader_args)

    # plot complete MC and Flow data
    plotutils.plot_training_setup(
        target_values=torch.tensor(X_train[:, :-1][X_train[:, -1] == 1]).float(),
        source_values=torch.tensor(X_train[:, :-1][X_train[:, -1] == 0]).float(),
        outdir=generator_path,
        dist_labels=["Flow generatd MC", "MC", "Transported"],
    )
    # sys.exit()
    classifier.run_training(
        train_dataloader_clf, valid_loader=valid_dataloader_clf, n_epochs=50
    )

    classifier.plot_auc(
        path=f"{generator_path}/clf_figures/clf_between_true_data_and_flow.png",
        epoch_nr="last",
    )

def test_conds_diff(path_clfs: str, titles: list = None, device: str = "cuda"):
    "Plotting the difference between conds sig/bkg estimators"
    if titles is None:
        titles = ["Pythia", "Herwig"]
    pt_range = torch.log(torch.linspace(20, 500, 501).view(-1, 1).to(device))
    plt.figure()
    for number, path in enumerate(path_clfs):
        ## load clf
        clf_config = misc.load_yaml(f"{path}/model_config.yml")
        clf_config["sigmoid"] = True
        clf_config["device"] = device
        classifier = DenseNet(**clf_config)
        latest_model = sorted(glob(f"{path}/models/*"), key=os.path.getmtime)[-1]
        try:
            classifier.load(latest_model, key_name="model_state_dict")
        except KeyError:
            classifier.load(latest_model, key_name="model")

        classifier = classifier.to(device)
        output = (
            classifier(pt_range)[1]  # pylint: disable=not-callable
            .cpu()
            .detach()
            .numpy()
        )

        plt.plot(pt_range.cpu().detach().numpy(), output, label=titles[number])
    plt.legend()

def train_bkg_subtraction(config, path_to_generator, mc:bool=False, sample_size:int=None,
                   **kwargs):
    bkg_calib = config.get("bkg_calib")
    save_path = f"{path_to_generator}/bkg_weighting/"
    
    # load ot bkg calib model
    if bkg_calib is not None:
        _, g_network = ot_utils.load_model_w_hydra(bkg_calib, -1)
    
    # load target data
    if any(['mc' in i.lower() for i in config['data']['paths']]):
        logging.info("Training the background subtraction required the target distribution - the paths has MC in it which might not be correct - EXITING!")
        sys.exit()
        
    #### DATA SETUP ####
    target_sample, scaler, weights, _ = get_samples(config['data'])

    # splitting data and creating sampler for weights

    # data = np.c_[data, np.ravel(weights)]
    
    # # Swap weight and labels
    # data[:, [-2, -1]] = data[:, [-1, -2]]

    if config['data'].get("maxevents") is not None:
        target_sample = target_sample[:config['data']["maxevents"]]

    
    if config['data'].get("use_weights") is not None:
        if sample_size is None:
            sample_size = len(target_sample)
        index = np.random.choice(len(target_sample), sample_size, p=weights/np.sum(weights))
        target_sample=target_sample[index]
    
    if len(target_sample)<5_000_000:
        index = np.random.choice(len(target_sample), 5_000_000)
        target_sample=target_sample[index]

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_path+"/plots/", exist_ok=True)

    # define target iterator
    # define source iterator using flow
    cvx_dims = len(config['data']['transportnames'])
    noncvx_dims = len(config['data']['condnames'])

    # create target iterator
    if not mc:
        target_sample = torch.tensor(target_sample).float()

        source = TemplateGenerator(
            path_to_generator,
            target_sample[:, :len(config['data']['condnames'])].cpu().clone(),
            generate_bkg=True,
            generate_sig=True,
            device=config["device"],
            batch_size=512,
            duplications=1,
            cvx_dim=cvx_dims,
            use_bkg_weighting=False,
        )
    else: # for flow to flow stuff
        mask_bjets = np.isin(target_sample[:, -1], [5])
        source = loader.Dataset(
            np.c_[target_sample[mask_bjets][:, :-1],
                  np.zeros(mask_bjets.sum())],
            batch_size=512,
            device=config["device"],
            cvx_dim=cvx_dims,
            noncvx_dim=noncvx_dims,
            # scaler=source.scaler
        )

    target = loader.Dataset(
        target_sample,
        batch_size=512,
        device=config["device"],
        cvx_dim=cvx_dims,
        noncvx_dim=noncvx_dims,
    )

    dist_styles=[
                {"label": "MC", "color": "black"},
                {"label": "MC bkg", "color": "red"},
                ]
    torch.cuda.empty_cache()

    source_data = source.data.detach().clone()
    source_data = source_data[source_data[:,-1]==0] # only bkg

    target_data = target.data.detach().clone()
    target_data[:, -1] = 1

    if bkg_calib is not None:
        
        # calib bkg
        source_data[:, 1:4] = g_network.chunk_transport(
            totransport = source_data[:, 1:4],
            conditionals = source_data[:, :1],
            n_chunks=200)

        # calculate dl1r
        dist_styles.append({"label": f'{dist_styles[-1]["label"]} + OT',
                            "color": "blue"})

    dataset = torch.cat([source_data, target_data], 0)

    X_train, X_test = train_test_split(
        dataset, test_size=0.20, shuffle=True
    )
    dataloader_style = {"pin_memory": False, "batch_size": 512, "drop_last": False,
                        "shuffle": True}
    train_loader = DataLoader(X_train,**dataloader_style)
    valid_loader = DataLoader(X_test,**dataloader_style)

    # setup density ratio estimator
    dense_net_args = config["model"]["dense"]
    dense_net_args["input_dim"] = cvx_dims+noncvx_dims
    dense_net_args["sigmoid"]=True
    dense_net_args['using_weights']=False

    ### run clf ###
    dense_net = DenseNet(**dense_net_args)
    dense_net.eval_str="valid_loss"
    
    dense_net.run_training(train_loader, valid_loader,
                           n_epochs=kwargs.get("n_epochs", 20), #config.dense_epochs,
                           standard_lr_scheduler=True, clip_bool=True)
    dense_net.plot_log(save_path)
    dense_net.save(
        f"{save_path}/bkg_down_weighting.pth"
        )
    misc.save_yaml(dense_net_args,
                f"{save_path}/model_config.yaml")

    ##### EVALUATE the background subtraction #####
    source_new = TemplateGenerator(
        path_to_generator,
        target_sample[:, :1].cpu().clone(),
        generate_bkg=True,
        device=config["device"],
        batch_size=512,
        duplications=1,
        cvx_dim=cvx_dims,
        use_bkg_weighting=False,
    )

    dataset_ny=[target_data.numpy()]

    ratio=[np.ones(len(target_data.numpy()))]
    source_new.classifier.device='cuda'
    dense_net.device='cpu'

    source_data = source_new.data.detach().clone()
    source_data[:,-1] = -source_data[:,-1] #used in sig_bkg_ratio

    for ch in source_data.chunk(10):
        # source_new.classifier classify if signal or background
        psig = source_new.classifier.cpu()(ch[:,:1].cpu())[-1]

        pbkg = 1-source_new.classifier.sigmoid_layer(psig).cpu().detach().numpy()
        # bkg_v_sig_ratio = ((1-source_new.classifier.cpu()(ch[:,:1].cpu())[-1]).cpu().detach().numpy())

        ratio.append(1/dense_net.cpu()(ch[:,:-1].cpu())[0].cpu().detach().numpy()*1/pbkg)

    dataset_ny.append(source_data.numpy())
    dataset_ny = np.concatenate(dataset_ny, 0)

    ratio = [np.ravel(i) for i in ratio]
    ratio = np.concatenate(ratio, 0)
    ratio[ratio>1]=1
    # xlabel=["pt","pb", "pc","pu"]

    log_pt = np.percentile(target_data[:, 0], [0, 25, 50, 75, 100])
    xlabels = config['data']['condnames']+config['data']['transportnames']
    for lpt, hpt in zip(log_pt[:-1], log_pt[1:]):
        for i in range(len(xlabels)):
            style={"bins":50, "histtype": "step"}#, "range":[-5,dl1r_cut]}
            mask_low_pt = (dataset_ny[:,0]<hpt) & (dataset_ny[:,0]>lpt)
            sig_bkg_ratio =np.isin(dataset_ny[:,-1][mask_low_pt], [0,-1]).sum()
            mask_sig = dataset_ny[:,-1]==1
            mask_bkg = dataset_ny[:,-1]==0

            bkg_ratio = ratio[mask_low_pt & mask_bkg]
            # bkg_ratio = bkg_ratio/len(ratio)
            n_events = np.sum(mask_low_pt & mask_sig)
            fig, ax = plt.subplots(1)
            counts, _= plot.plot_hist(
                dataset_ny[:,i][mask_low_pt & mask_sig],
                dataset_ny[:,i][mask_low_pt & mask_bkg],
                dataset_ny[:,i][mask_low_pt & mask_bkg],
                weights=[np.ones(n_events)/n_events,
                        np.ravel(bkg_ratio)/sig_bkg_ratio,
                        np.ones_like(bkg_ratio)/sig_bkg_ratio],
                style=style,
                dist_styles=[{"label":"Data", "color": "blue"},
                            {"label": "New BKG", "color": "green"},
                            {"label":"Old BKG", "color": "red"}],
                normalise=False,
                legend_kwargs={"loc":"upper left",
                                "title": f"pT [{np.round(np.exp(lpt),3)}, {np.round(np.exp(hpt),3)}]"},
                ax=ax
                        )
            ax.set_xlabel(xlabels[i])
            ax.set_ylim([5e-4, 0.15])
            ax.set_yscale("log")
            os.makedirs(f"{save_path}/plots/{xlabels[i]}/", exist_ok=True)
            
            misc.save_fig(fig, f"{save_path}/plots/{xlabels[i]}/{lpt}_{hpt}.pdf")
    return save_path