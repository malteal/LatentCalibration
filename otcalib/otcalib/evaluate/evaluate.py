"Show evaluation of the saved models"

import json
from glob import glob
from itertools import product

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch as T
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


def gif(imgs_paths, outdir_name):
    imgs = (Image.open(f) for f in imgs_paths)
    img = next(imgs)  # extract first image from iterator
    if not ".gif" in outdir_name:
        outdir_name += ".gif"
    img.save(
        fp=outdir_name,
        format="GIF",
        append_images=imgs,
        save_all=True,
        duration=110,
        loop=0,
    )


def create_gif(folder_path, plot_type, every=None, max_frames=None):
    # filepaths
    path = f"{folder_path}/plots/{plot_type}"
    fp_out = f"{'/'.join(path.split('/')[:-2])}/{plot_type.split('/')[0]}.gif"
    # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
    paths = glob(path)[::every]
    # print(paths)
    paths = np.array(paths)[
        np.argsort([int(i.split("valid")[-1].split("_")[1]) for i in paths])
    ]
    paths = paths[:max_frames]
    gif(imgs_paths=paths, outdir_name=fp_out)


def get_values_out_of_dict(
    keys, model_loss, diff
):  # TODO write better function for unpacking dicts!
    if isinstance(keys, str):
        values = model_loss[keys]
    elif diff and (isinstance(keys, list)) and (len(keys) == 2):
        values = np.abs(model_loss[keys[0]]) - np.abs(model_loss[keys[1]])
    elif (isinstance(keys, list)) and (len(keys) == 2):
        if isinstance(keys[1], list):
            values = np.mean([model_loss[keys[0]][i] for i in keys[1]], 0)
        else:
            values = model_loss[keys[0]][keys[1]]
    elif (isinstance(keys, list)) and (len(keys) == 3):
        values = model_loss[keys[0]][keys[1]][keys[2]]
    else:
        raise NameError(
            f"Keys cannot be found: {keys} - Possible keys: {model_loss.keys()}"
        )
    return values


def plot_loss(
    file_path: str,
    previous_fig: plt.Figure = False,
    keys: list = None,
    style: dict = None,
    diff: bool = False,
    skip: int = None,
    function: callable = None,
    error_keys: list = None,
):
    """Visulize metrics from the loss file

    Parameters
    ----------
    file_path : str
        Path to the metric file
    previous_fig : plt.Figure, optional
        plt.Figure, used to plot ontop of figures, by default False
    fig_color : str, optional
        colors in the figure, by default "blue"
    style : dict, optional
        style for the plt.figure, by default None
    diff : list, optional
        if the values should be subtracted, by default None

    Returns
    -------
    plt.Figure
        output the figure with new data on it
    """
    if not previous_fig:
        previous_fig = plt.figure()
    with open(file_path + "/log.json", "r+", encoding="utf8") as fconfig:
        model_loss = json.load(fconfig)
    if isinstance(keys, list):
        keys[0] = [i for i in model_loss.keys() if keys[0] in i][0]
    # try:
    data = get_values_out_of_dict(keys, model_loss, diff)
    if len(data) == 0:
        return previous_fig, model_loss
    data = np.nan_to_num(data, nan=999, posinf=999, neginf=999)
    if function is not None:
        try:
            data = function(data, 1)
        except (ValueError, TypeError):
            data = function(data)
    colors = list(mcolors.BASE_COLORS.keys())[::-1][1:]
    for col in range(1 if len(data.shape) == 1 else data.shape[1]):
        if len(data.shape) == 1:
            values = data
        else:
            style["color"] = colors[col]
            values = data[:, col]
        if error_keys is None:
            if keys[1] in ["ks_values"]:
                style["label"] += f"{np.argmax(values)}: {round(np.max(values),4)}"
            else:
                if (len(keys) == 2) or (len(keys) > 8):
                    ke = ""
                else:
                    ke = keys
                style["label"] += f"{ke} {np.argmin(values)}: {round(np.min(values),4)}"
        if isinstance(skip, list):
            plt.plot(
                np.linspace(
                    skip[0],
                    skip[1] if skip[1] < len(values) else len(values),
                    len(values[skip[0] : skip[1]]),
                ),
                values[skip[0] : skip[1]],
                **style,
            )
        else:
            if error_keys is None:
                plt.plot(values[skip:], **style)
            else:
                skip = skip if skip is not None else 0
                values = np.array(values)
                yerr = np.array(get_values_out_of_dict(error_keys, model_loss, diff))
                mask = (values[skip:]) / yerr > 2.5
                x_values = np.arange(skip, len(values))[mask]
                style["label"] += (
                    f"{x_values[np.min(values[mask]) == values[mask]]}:"
                    f" {round(np.min(values[mask]),4)}"
                )
                plt.errorbar(x_values, values[skip:][mask], yerr=yerr[mask], **style)
    # except KeyError as error_value:
    #     print(f"{error_value}: model loss properly missing the key")
    plt.xlabel("Epochs")
    # plt.legend()
    return previous_fig, model_loss


def run_clf_evaluation(discriminator, discriminator_str, data, **kwargs):
    metric = kwargs.get("metric", {})
    first_training = kwargs.get("first_training", False)
    # discriminator evaluation - matching conds
    keys_in_eval = data.keys()
    if "total" in data:
        keys_in_eval = ["total"]

    # create data for discriminator
    disc_data = T.tensor([])
    disc_transport = T.tensor([])
    
    # unpack dict of data to create the data for the discriminator
    for conds_val in keys_in_eval:

        # target distribution
        input_data = T.concat(
            [
                data[conds_val]["truth"]["conds"],
                data[conds_val]["truth"]["transport"],
                T.ones(len(data[conds_val]["truth"]["conds"]), 1),
            ],
            1,
        )
        disc_data = T.concat([disc_data, input_data], 0)

        # transport distribution 
        input_data = T.concat(
            [
                data[conds_val][discriminator_str]["conds"],
                data[conds_val][discriminator_str]["eval_transport"],
                T.zeros(len(data[conds_val][discriminator_str]["conds"]), 1),
            ],
            1,
        )
        disc_transport = T.concat([disc_transport, input_data], 0)

    # detach data
    disc_transport = disc_transport.detach()
    disc_data = disc_data.detach()

    # have to use random state to compare conds
    # split data
    train_data, test_data = train_test_split(disc_data, test_size=0.25, random_state=42)
    train_trans, test_trans = train_test_split(
        disc_transport, test_size=0.25, random_state=42
    )
    
    # the distribution is combined this way to fix the condition
    # so they are the same for each batch
    train = T.ones((2 * len(train_data), train_data.shape[1]))
    test = T.ones((2 * len(test_data), test_data.shape[1]))

    # shuffle idx so each pair of the target and transport distribution might be the same
    # but the order is random between each pair is random
    idx = T.randperm(len(train_data))
    train[::2] = train_data[idx]
    train[1::2] = train_trans[idx]    
    
    # doesnt matter for the test set
    test[::2] = test_data
    test[1::2] = test_trans

    # create dataloaders
    dataloader_args = {"batch_size": 512, "drop_last": False, "shuffle": False}
    train_dataloader_clf = DataLoader(train.detach(), **dataloader_args)
    valid_dataloader_clf = DataLoader(test.detach(), **dataloader_args)

    # init new log file in classifier
    discriminator.init_new_log_dict()

    # init new optimiser
    discriminator.set_optimizer(True, lr=1e-4 if first_training else 1e-5)

    # train classifier
    discriminator.run_training(
        train_dataloader_clf,
        valid_loader=valid_dataloader_clf,
        n_epochs=200 if first_training else 5,
        verbose=first_training,
        standard_lr_scheduler = first_training
    )

    # logging classifier performance
    metric["AUC"] = np.max(discriminator.loss_data["valid_auc"])

    metric["train_AUC"] = discriminator.evaluate(train_dataloader_clf)

    # test as a function of conditions
    if data[conds_val]["truth"]["conds"].shape[1] > 0:
        # ONLY FOR 1D CONDS or in 1d slice

        conds_quantile = T.quantile(
            test[:, 0], kwargs.get("conds_range", T.linspace(0, 1, 7))
        )

        conds_dist = test[:, 0]
        conds_quantile = np.round(conds_quantile, 2)
        for low, high in zip(conds_quantile[:-1], conds_quantile[1:]):
            mask = (conds_dist >= low) & (conds_dist < high)
            valid_dataloader_clf = DataLoader(test[mask].detach(), **dataloader_args)
            auc = discriminator.evaluate(valid_dataloader_clf)
            metric[f"AUC_{low.numpy():.2f}_{high.numpy():.2f}"] = auc

    return metric


def comparing_metrics(
    paths: list,
    keys: list,
    diff: bool = False,
    same: bool = False,
    skip: int = None,
    function: callable = None,
    error_keys: list = None,
    **kwargs,
):
    """Testing performance of model

    Parameters
    ----------
    paths : list
        list of paths containing performance info
    keys : list
        each path should point to a dict - keys are the keys selected in the dict
    diff : bool, optional
        if the values within the dict should be subtracted, by default False
    same : bool, optional
        if the values in the dict should be subtracted, by default False

    Returns
    -------
    dict
        Return the latest dict
    """
    assert isinstance(paths, list), "paths should be a list"

    colors = ["blue", "red", "green", "black", "orange"]
    linestyles = ["-", "--", "-.", ":"]
    colors_and_styles = product(linestyles, colors)
    if same:
        fig_loss = plt.figure(figsize=(6, 5))
    for key in keys:  # zip(linestyles, keys): marker
        if not same:
            fig_loss = plt.figure(figsize=(6, 5))

        for _, (path, color_and_style) in enumerate(zip(paths, colors_and_styles)):
            style = {
                "linestyle": color_and_style[0],
                "label": "_".join(path[:-1].split("/")[-1].split("_")[:7]),
                "color": color_and_style[1],
            }
            if "key_in_label" in kwargs:
                style["label"] = key
            fig_loss, performance_values = plot_loss(
                file_path=path,
                previous_fig=fig_loss,
                keys=key.copy() if isinstance(key, list) else key,
                style=style,
                diff=diff,
                skip=skip,
                function=function,
                error_keys=error_keys,
            )
        # if fig_loss.legends != []:
        if (len(paths) <= 3) or kwargs.get("legend_bool", False):
            plt.legend()
        plt.title(key)
    # fig_loss.tight_layout()
    return performance_values, fig_loss
