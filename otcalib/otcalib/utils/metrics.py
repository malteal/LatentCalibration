"This script define metrics used to valid performance"
import numpy as np
import torch
from scipy import stats
from scipy.spatial import distance

from utils.transformations import numpy_check

# from geomloss import SamplesLoss


class Metrics:
    """Class that define the metric"""

    def __init__(
        self,
        geomloss_input: list,
    ):
        """Init metric

        Parameters
        ----------
        geomloss_input : list
            input for GeomLoss
        """
        self.geomloss_input = geomloss_input
        self.geomloss = []
        self.keys = ["ks_values", "wasserstein", "jensenshannon"]
        for geo_input in self.geomloss_input:
            self.keys.append(geo_input["loss"])
            # self.geomloss.append(self.geomloss_nd(geo_input))

    # def geomloss_nd(self, geo_input:dict):
    #     """https://www.kernel-operations.io/geomloss/api/pytorch-api.html"""
    #     return SamplesLoss(**geo_input)
    @staticmethod
    def ks_test1d(source: torch.Tensor, target: torch.Tensor) -> list:
        """ks_test in 1d

        Parameters
        ----------
        source : torch.Tensor
            Source distribution
        target : torch.Tensor
            target distribution

        Returns
        -------
        list
            Return the ks probability between the two distributions
            divided in 1d distributions
        """
        ks_values = []
        source = numpy_check(source)
        target = numpy_check(target)
        for col in range(source.shape[1]):
            ks_values.append(stats.ks_2samp(source[:, col], target[:, col]).pvalue)
        return np.float64(ks_values[0])

    @staticmethod
    def jensenshannon(source: torch.Tensor, target: torch.Tensor) -> np.float64:
        """Run the KL-divergence - not sure if it is implemented correctly

        Parameters
        ----------
        source : torch.Tensor
            Source distribution
        target : torch.Tensor
            Target distribution

        Returns
        -------
        np.float64
            kl div between the distributions
        """
        # https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html

        source = numpy_check(source)
        target = numpy_check(target)

        xmin, xmax = np.min(target), np.max(target)
        style = {
            "range": [[xmin, xmax]] * target.shape[1],
            "bins": 50,
        }
        source, _ = np.histogramdd(source, **style)
        target, _ = np.histogramdd(target, **style)

        source_counts_norm = source / np.sum(source)
        target_counts_norm = target / np.sum(target)

        for i in [distance.jensenshannon]:  # ready if more metric should be added
            jensenshannon = i(source_counts_norm, target_counts_norm)
        jensenshannon = np.nan_to_num(jensenshannon, 0)
        return np.float64(np.sum(jensenshannon))

    @staticmethod
    def nd_wasserstein_distance(
        source: torch.Tensor, target: torch.Tensor
    ) -> np.float64:
        """Runs 1d wasserstein on each column in source/target

        Parameters
        ----------
        source : torch.Tensor
            Source distribution
        target : torch.Tensor
            target distribution

        Returns
        -------
        np.float64
            wasserstein distance in each dimension
        """
        idx_source = np.random.choice(np.arange(0, len(source), 1), len(target))
        source = numpy_check(source)[idx_source]
        target = numpy_check(target)
        wasser = np.mean(np.abs(np.sort(source.T) - np.sort(target.T)), 1)
        if len(wasser) == 1:
            return np.float64(wasser)

        return list(np.float64(wasser))

    @staticmethod
    def get_keys(bins):
        keys = []
        for name in bins.keys():
            for i, j in zip(bins[name][:-1], bins[name][1:]):
                for k in ["ks_values", "wasserstein", "jensenshannon"]:
                    keys.append(k + f"{name}_{i}_{j}")
        return keys

    def run(  # pylint: disable=dangerous-default-value, too-many-arguments
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        geomloss_size: int = None,
        append: dict = {},
        sub_dict: str = "",
        column_name: str = "",
    ) -> dict:
        """Runs all the metrics and add it to a dict to be save in a json file

        Parameters
        ----------
        source : torch.Tensor
            source distribution
        target : torch.Tensor
            target distribution
        geomloss_size : int, optional
            How many point should be used in the geomloss metrics, by default None
        append : dict, optional
            The dict where the metric values is saved in, by default {}
        sub_dict : str, optional
            sub dict of the append dict, by default ""

        Returns
        -------
        dict
            output a dict with the metric values
        """
        if isinstance(source, list):
            source = np.array(source)
            target = np.array(target)
            source.shape = (len(source), 1)
            target.shape = (len(target), 1)
        if sub_dict == "":
            append["ks_values"] = self.ks_test1d(source, target)
            append["wasserstein"] = self.nd_wasserstein_distance(source, target)
            append["jensenshannon"] = self.jensenshannon(source, target)
            for geo_input, loss in zip(self.geomloss_input, self.geomloss):
                append[geo_input["loss"]] = (
                    loss(source[:geomloss_size], target[:geomloss_size])
                    .detach()
                    .numpy()
                )
        else:
            # create keys if they do not exists
            for key in np.array(self.keys):
                if not key + column_name in list(append[sub_dict].keys()):
                    append[sub_dict][key + column_name] = 999
            # fill keys
            append[sub_dict]["ks_values" + column_name] = self.ks_test1d(source, target)
            append[sub_dict]["wasserstein" + column_name] = (
                self.nd_wasserstein_distance(source, target)
            )
            if source.shape[1] < 3:
                append[sub_dict]["jensenshannon" + column_name] = self.jensenshannon(
                    source, target
                )
            for geo_input, loss in zip(self.geomloss_input, self.geomloss):
                append[sub_dict][geo_input["loss"]].append(
                    np.float64(
                        loss(source[:geomloss_size].cpu(), target[:geomloss_size].cpu())
                        .detach()
                        .numpy()
                    )
                )
        return append
