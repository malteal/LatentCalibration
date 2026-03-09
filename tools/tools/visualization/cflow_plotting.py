"visualization of conditional flow performance"

from multiprocessing.sharedctypes import Value
import matplotlib.pyplot as plt
import numpy as np
import torch 
from tqdm import tqdm
from nflows.transforms.base import InputOutsideDomain

np.seterr(invalid='ignore')

class Plotting:
    def __init__(
        self,
        flow=None,
        test_sample=None,
        test_sample_conds=None,
        n_times_sample:int=1,
        weights=None,
        training_conds=None,
        scaler=None,
        device="cuda",
        **kwargs,
    ):
        self.scaler=scaler


        self.test_sample = test_sample.cpu().detach().numpy()
        self.test_sample_conds = test_sample_conds.cpu()

        # transform test_sample
        if (self.scaler is not None) & kwargs.get("run_scaler", True):
            self._normalize()
        elif kwargs.get("inverse_scale", False): # inverse trans target
            self.test_sample = self.scaler.inverse_transform(
                np.c_[self.test_sample_conds.detach().numpy(), self.test_sample]
                )[:,test_sample_conds.shape[1]:]

        self.training_conds = training_conds
        self.n_times_sample = n_times_sample
        self.percentile_lst = kwargs.get("percentile_lst", [0.1, 99.9])

        if isinstance(training_conds, torch.Tensor):
            self.training_conds = self.training_conds.cpu().detach().numpy()

        self.weights = weights if weights is not None else np.ones_like(self.test_sample[:,0])
        if len(self.weights) != len(self.test_sample):
            raise ValueError("weights and test_sample have to have the same length")


        self.test_sample_conds = torch.concat([self.test_sample_conds for _ in range(self.n_times_sample)],0)

        self.sample_weights = np.ravel(np.concatenate([self.weights for _ in range(self.n_times_sample)],0))

        self.flow = flow
        self.device=device
        self._marginal_data = None

        self.ylim = kwargs["ylim"] if "ylim" in kwargs.keys() else [0.75,1.25]

        self.style = {"bins": 80}

        if any(["style" in i for i in kwargs.keys()]):
            self.style.update(kwargs["style"])

        self.style_error={"marker": "o", "linestyle": 'none', "markersize": 0,
                          "alpha":0.5}
        if flow is not None:
            self.update_sample()

    def _renormalize(self) -> None:
        try:
            self.sample = self.scaler.inverse_transform(torch.tensor(self.sample))
            # self.sample = utils.logit(utils.probsfromlogits(self.sample))
        except ValueError:
            self.sample  = self.scaler.inverse_transform(torch.concat([self.test_sample_conds, torch.tensor(self.sample)], 1))[:,self.test_sample_conds.shape[1]:]

    def _normalize(self) -> None:
        try:
            self.test_sample = torch.tensor(self.scaler.transform(self.test_sample)).float().numpy()
            # pass
        except ValueError:
            test_all = self.scaler.transform(np.c_[self.test_sample_conds, self.test_sample])
            self.test_sample_conds = torch.tensor(test_all[:,:self.test_sample_conds.shape[1]]).float()

    def update_sample(self, flow=None) -> None:
        if flow is not None:
            self.flow = flow
        # sample data
        self.sample=[]

        n_chunks_size = 50_000
        if len(self.test_sample_conds) < n_chunks_size:
            n_chunks_size = len(self.test_sample_conds)
        with torch.no_grad():

            n_chunks = self.test_sample_conds.shape[0]//n_chunks_size
            
            sample=[]
            for i in self.test_sample_conds.chunk(n_chunks):
                try:
                    sample.append(
                        self.flow.sample(1, i.to(self.device))[:,0,:None].cpu().detach().numpy()
                        )
                except InputOutsideDomain:
                    sample.append(
                        self.flow.sample(1, i.to(self.device))[:,0,:None].cpu().detach().numpy()
                        )

            # sample.extend(np.concatenate(sample,0))

        self.sample = np.concatenate(sample,0)

        if self.scaler is not None:
            self._renormalize()

    def probability_2d(self, title="") -> None:
        fig, ax = plt.subplots(1, 2)
        ax[0].contourf(self.xgrid.numpy(), self.ygrid.numpy(), self.zgrid0.numpy())
        ax[1].contourf(self.xgrid.numpy(), self.ygrid.numpy(), self.zgrid1.numpy())
        plt.title(title)
        plt.show()

    def plot_marginals(self, marginal_data=None,
                       fig=None, ax_1=None, ax_2=None,
                       plot_truth=True, **kwargs):

        n_of_var = len([i for i in self._marginal_data.keys() if "bins" in i])
        if marginal_data is None:
            marginal_data = self._marginal_data.copy()
        if fig is None:
            fig, (ax_1, ax_2) = plt.subplots(
                                            2, n_of_var, gridspec_kw={"height_ratios": [3, 1]},
                                            figsize=(n_of_var*8,6), sharey=False, sharex="col",
                                            squeeze=False
                                        )
        if not isinstance(ax_1, (list, np.ndarray)) or not isinstance(ax_2, (list, np.ndarray)):
            raise TypeError("ax_1 & ax_2 has to be a single dim list - like [AxesPlot]")
        # plot truth
        margin_style = self.style.copy()
        margin_style_error = self.style_error.copy()
        for nr in range(n_of_var):
            bins = self._marginal_data[f"bins_{nr}"]
            margin_style["bins"] = bins
            
            margin_style_error["x"] = 0.5*(bins[1:]+bins[:-1])
            # if True:
            margin_style_error["xerr"] = np.abs(bins[:-1]-bins[1:])/2
            # plot truth ratio line
            ax_2[nr].plot(bins, np.ones_like(bins), color="black", alpha=0.5)

            # plot truth
            ax_1[nr] = self.plot_errorbar(bins, marginal_data["Truth"][f"counts_{nr}"],
                                          marginal_data["Truth"][f"std_{nr}"],
                                          ax=ax_1[nr],
                                          label=kwargs.get("truth_label", "Truth"),
                                          color="black",
                                          weights=marginal_data["Truth"]["weights"],
                                          )
            if marginal_data["Truth"][f"std_{nr}"] is not None:
                ax_1[nr].fill_between(margin_style_error["x"],
                                    ((marginal_data["Truth"][f"counts_{nr}"]
                                    +marginal_data["Truth"][f"std_{nr}"])/
                                    np.sum(marginal_data["Truth"]["weights"])),
                                    ((marginal_data["Truth"][f"counts_{nr}"]
                                    -marginal_data["Truth"][f"std_{nr}"])/
                                    np.sum(marginal_data["Truth"]["weights"])),
                                    color="black",
                                    alpha=0.1, interpolate=True, step = 'mid',)

                memStd_truth_ratio_upper = np.nan_to_num(((marginal_data["Truth"][f"counts_{nr}"]
                                                        +marginal_data["Truth"][f"std_{nr}"])
                                                        /marginal_data["Truth"][f"counts_{nr}"]), 0)

                memStd_truth_ratio_lower = np.nan_to_num(((marginal_data["Truth"][f"counts_{nr}"]
                                                        -marginal_data["Truth"][f"std_{nr}"])
                                                        /marginal_data["Truth"][f"counts_{nr}"]), 0)

                ax_2[nr].fill_between(margin_style_error["x"], memStd_truth_ratio_upper,
                                        memStd_truth_ratio_lower, color="black",
                                        alpha=0.1, interpolate=True, step = 'mid')
        if "color" in kwargs:
            colors = kwargs["color"]
        else:
            colors = ["blue", "red", "green", "darkorange"]
        labels = kwargs.get("labels", None)
        nr_type=0
        for (name,value) in marginal_data.items():
            if ("bins" in name) or ("Truth" in name):
                continue
            for nr in range(n_of_var):
                bins = self._marginal_data[f"bins_{nr}"]
                margin_style["bins"] = bins
                
                margin_style_error["x"] = 0.5*(bins[1:]+bins[:-1])
                # if True:
                margin_style_error["xerr"] = np.abs(bins[:-1]-bins[1:])/2

                # plot different sample hist

                ax_1[nr] = self.plot_errorbar(bins, value[f"counts_{nr}"],
                                                value[f"std_{nr}"],
                                                color=colors[nr_type],
                                                label=name if labels is None else labels[nr_type],
                                                ax=ax_1[nr],
                                                weights=value["weights"])

                # plot different sample ratio plot
                ax_2[nr] = self.ratio_plot(value[f"counts_{nr}"],
                                           marginal_data["Truth"][f"counts_{nr}"],
                                           value[f"std_{nr}"],
                                           ax=ax_2[nr], color=colors[nr_type],
                                           style=margin_style_error)
                if "xlabel" in kwargs:
                    ax_2[nr].set_xlabel(kwargs["xlabel"][nr])
            nr_type+=1

        # sys.exit()
        if plot_truth: # show only legend single time
            for i in range(len(ax_1)):
                ax_1[i].legend(frameon=False, loc="best",
                               title=kwargs.get("legend_title", None))
        ax_1[0].set_ylabel("Normalised entries")
            

        for i in range(n_of_var):
            if kwargs.get("y_scale", None) is not None:
                ax_1[i].set_yscale(kwargs["y_scale"])
                ax_1[i].set_ylim([1e-6, ax_1[i].get_ylim()[1]])
            else:
                ax_1[i].set_ylim([0, ax_1[i].get_ylim()[1]])
            ax_2[i].set_ylim(self.ylim)
            ax_2[i].set_xlim([self._marginal_data[f"bins_{i}"].min(),
                            self._marginal_data[f"bins_{i}"].max()])

        plt.tight_layout()
        return fig,ax_1,ax_2

    @staticmethod
    def plot_errorbar(bins, y, menStd, ax, label, color, weights=None):
        if weights is not None:
            y = y/np.sum(weights)
            menStd = menStd/np.sum(weights)
        
        bincenters = 0.5 * (bins[1:] + bins[:-1])
        ax.errorbar(
            bincenters,
            y,
            yerr=menStd,
            xerr=np.abs(bins[:-1]-bins[1:])/2,
            markersize=0,
            color=color,
            label=label,
            linestyle="none",
            alpha=0.7,
        )
        return ax

    @staticmethod
    def bin_error(data, bins, weights=None):

        if weights is None:
            weights = np.ones_like(data)

        menStd = np.sqrt(np.histogram(data, bins=bins, weights=np.array(weights)**2)[0])

        return menStd

    def marginals(self, additional_data=None, additional_names=None, **kwargs):
        # print("calculating marginals")
        #scale back to original distribution
        if additional_names is None:
            additional_names = []

        store_data={i:{} for i in ["Flow", "Truth"]}
        margin_style = self.style.copy()
        for nr in range(self.sample.shape[1]):

            if ("bins" in kwargs) & bool((kwargs.get("bins", False))):
                margin_style["bins"] = kwargs["bins"][nr]

            if "range" in kwargs:
                margin_style["range"] = kwargs["range"][nr]

            elif isinstance(margin_style["bins"], int) and (margin_style.get("range") is None): # there might be more ints...
                margin_style['range'] = np.percentile(self.sample[:,nr], self.percentile_lst)

            
            counts_sample, bins= np.histogram(self.sample[:,nr],
                                              weights = self.sample_weights,
                                            #   weights=sample_weights,
                                              **margin_style)
            
            counts_truth,_= np.histogram(self.test_sample[:,nr], weights=self.weights, **margin_style)

            # calculate the histo error
            menStd_truth = self.bin_error(self.test_sample[:,nr], bins)
            menStd_flow = self.bin_error(self.sample[:,nr], bins)

            store_data[f"bins_{nr}"] = bins

            store_data["Truth"][f"std_{nr}"] = menStd_truth
            store_data["Flow"][f"std_{nr}"] = menStd_flow

            store_data["Flow"][f"counts_{nr}"] = counts_sample
            store_data["Truth"][f"counts_{nr}"] = counts_truth

            store_data["Flow"]["weights"] = np.ones_like(self.sample[:,nr])
            store_data["Truth"]["weights"] = np.ones_like(self.test_sample[:,nr])

            if additional_data is not None:
                for _nr, (j, data) in enumerate(zip(additional_names, additional_data)):
                    # if kwargs.get("clip", False):
                    #     data[:,nr] = np.clip(data[:,nr], *margin_style["range"])
                    if j not in store_data: store_data[j]={}
                    counts_out, _ = np.histogram(data[:,nr],
                                                weights= kwargs["weights"][_nr],
                                                **margin_style)
                    menStd = self.bin_error(data[:,nr], bins,
                                            weights=kwargs["weights"][_nr])
                    store_data[j][f"std_{nr}"] = menStd
                    store_data[j][f"counts_{nr}"] = counts_out
                    store_data[j]["weights"] = kwargs["weights"][_nr]
                    
            if "range" not in self.style:
                margin_style.pop("range", None)

        self._marginal_data = store_data
        return store_data

    def uncertainty_diff(self):
        fig, ax_lst = plt.subplots(1, 3, figsize=(12, 6))
        for ax, val, name in zip(
            ax_lst, np.arange(-1, 2, 1), ["Lower", "Central", "Upper"]
        ):

            upper_difference_flow = (
                self._marginal_data["Flow_counts_0"]
                + val * self._marginal_data["std_flow_0"]
            ) - (
                self._marginal_data["truth_counts_0"]
                + val * self._marginal_data["std_truth_0"]
            )
            upper_difference_reweight = (
                self._marginal_data["Reweight_counts_0"]
                + val * self._marginal_data["Reweight_std_0"]
            ) - (
                self._marginal_data["truth_counts_0"]
                + val * self._marginal_data["std_truth_0"]
            )

            style = {"bins": 30, "histtype": "step"}
            ax.set_title(name)
            _, bins, _ = ax.hist(
                upper_difference_reweight, label="Reweighting", color="red", **style
            )
            style["bins"] = bins
            _, bins, _ = ax.hist(
                upper_difference_flow, label="Flow", color="blue", **style
            )
        plt.legend(frameon=False)
        plt.tight_layout()
        return fig

    def pull_residual(self, style = None, xlabel=None, **kwargs):
        fig, ax = plt.subplots(1,2, figsize = (12,7), sharey=True)
        if style is None:
            style = {"bins": 25, "histtype": "step"}
        col_name = [i for i in self._marginal_data if (not "Truth" in i)
                                                    and (not "bins" in i) ]
        colors=kwargs.setdefault("colors", ["blue", "red", "black", "orange"])
        for i in range(2):
            for nr, col in enumerate(np.unique(col_name)):
                values = ((self._marginal_data[col][f"counts_{i}"]-
                        self._marginal_data["Truth"][f"counts_{i}"])/
                        (self._marginal_data[col][f"std_{i}"]))
                values = values[~np.isnan(values) & ~np.isinf(values)]
                mean = str(round(np.mean(values),2))
                std = str(round(np.std(values),2))
                label = rf"{col} $\mu$/$\sigma$: {mean}/{std}"
                counts, bins, _ = ax[i].hist(values,
                                            color=colors[nr], label=label, **style)

            ax[i].set_xlabel(rf"$c_{i}$" if xlabel is None else xlabel[i])
            # Shrink current axis's height by 10% on the bottom
            # Put a legend below current axis
            ax[i].legend(bbox_to_anchor=(0, 1.3, 1, 0.2),
                         loc="upper left",
                         frameon=False
                        #  mode="expand",
                        #  borderaxespad=0,
                        #  ncol=3
                         )
        ax[0].set_ylabel("Normalised entries")
        # plt.legend(frameon=False)
        # plt.tight_layout()
        return fig

    def ratio_plot(self, counts, counts_truth, menStd, ax, color, style):
        if counts_truth.sum() < 1.5: # if truth is normalized not completely correct
            counts_truth = counts_truth*counts.sum()
        else:
            counts_truth = counts_truth*counts.sum()/counts_truth.sum()
        y_counts = np.nan_to_num(counts/counts_truth,0)
        yerr_relative_upper = np.nan_to_num(menStd/counts_truth,0)

        mask_down = self.ylim[0]>=y_counts
        mask_up = self.ylim[1]<=y_counts

        # yerr_lower[mask_down | mask_up] = 0
        # yerr_relative_upper[mask_down | mask_up] = 0
        # y_counts[mask_down | mask_up] = -1
        ax.errorbar(y=y_counts, yerr=yerr_relative_upper,
                        color=color, **style
                        )
        #marker up
        ax.scatter(style["x"][mask_up],
                    np.ones(mask_up.sum())*(self.ylim[1]-self.ylim[1]/50),
                        color=color,marker='^',s=25, alpha=1)
        
        #marker down
        ax.scatter(style["x"][mask_down],
                  np.ones(mask_down.sum())*(self.ylim[0]+self.ylim[0]/30),
                            color=color,marker='v',s=25, alpha=1)
        return ax

    def plot_conds(self):
        fig, ax = plt.subplots(1, self.test_sample_conds.shape[1], figsize = (9,6), sharey=True)
        
        if self.test_sample_conds.shape[1] == 1:
            ax = [ax]
        conds_dist =self.test_sample_conds.cpu().detach().numpy()
        for nr in range(conds_dist.shape[1]):
            if self.training_conds is not None:
                ax[nr].hist(self.training_conds[:,nr],
                            color="red",
                            label="Training condition",
                            **self.style)
            else:
                self.style["range"] = [conds_dist[:,nr].min(),
                                    conds_dist[:,nr].max()]
                                
            ax[nr].hist(conds_dist[:,nr], color="green", label="Condition",
                        **self.style)

            ax[nr].set_title("Conditional distribution")
        plt.legend(frameon=False)
        return fig, ax
        
    @staticmethod
    def ratio_error(sample, truth, yerr=None, yerr_truth=None):
        if yerr is not None:
            return np.sqrt((yerr/truth)**2+(yerr_truth*sample/truth**2)**2)
        else:
            return np.sqrt((np.sqrt(sample)/truth)**2+(np.sqrt(truth)*sample/truth**2)**2)


    @staticmethod
    def probfromlogit(logits):
        # logitps = numpy_check(logitps)

        ps_value = 1.0 / (1.0 + np.exp(-logits))
        norm = np.sum(ps_value, axis=1)
        norm = np.stack([norm] * logits.shape[1]).T
        return ps_value / norm

    def plot_dl1r(self, labels_order, fc = 0.018):
        order_pb = np.argmax(["pb" in i.replace("_", "").replace("{", "") for i in labels_order])
        order_pu = np.argmax(["pu" in i.replace("_", "").replace("{", "") for i in labels_order])
        order_pc = np.argmax(["pc" in i.replace("_", "").replace("{", "") for i in labels_order])
        test_sample = self.test_sample
        sample = self.sample

        truth_sample = self.probfromlogit(test_sample)
        sample = self.probfromlogit(sample)
        style = {"bins": 80, "histtype": "step", "density": False, "alpha":0.5}

        pb_value_truth, pc_value_truth, pu_value_truth = (truth_sample[:,order_pb],
                                                        truth_sample[:,order_pc],
                                                        truth_sample[:,order_pu])
        pb_value, pc_value, pu_value = (sample[:,order_pb],
                                        sample[:,order_pc],
                                        sample[:,order_pu])
        dl1r_value = np.log(pb_value) - np.log(fc * pc_value + (1-fc) * pu_value)
        dl1r_value_truth = np.log(pb_value_truth) - np.log(fc * pc_value_truth + (1-fc) * pu_value_truth)

        # plotting the dl1r
        fig, ax = plt.subplots(2, 1, gridspec_kw={"height_ratios": [3, 1]}, 
                    figsize=(9,6), sharex=True
                )
        style["range"] = [np.percentile(dl1r_value_truth, 0.01),
                               np.percentile(dl1r_value_truth, 99.99)]
        counts, bins, _ = ax[0].hist(dl1r_value, color="blue", label = "Flow sample", **style)
        counts_truth, _, _ = ax[0].hist(dl1r_value_truth, color="black", label = "Truth", **style)

        self.style_error["x"] = 0.5*(bins[1:]+bins[:-1])
        self.style_error["xerr"] = np.abs(bins[:-1]-bins[1:])/2
        ax[1].errorbar(y=counts/counts_truth, color="blue",
                        yerr=self.ratio_error(counts, counts_truth),
                        **self.style_error)
        ax[1].plot(bins, np.ones_like(bins), "k--")

        ax[0].set_ylabel("#")
        ax[1].set_ylim([0.8, 1.2])
        ax[1].set_ylabel("ratio")
        ax[1].set_xlabel("DL1r")
        ax[0].legend(frameon=False)
        return fig