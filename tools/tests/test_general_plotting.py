from tools.visualization import general_plotting as plot
import matplotlib
import numpy as np

np.seterr(invalid='ignore')

class TestGeneralPlotting:
    # simple init
    arr = np.random.normal(0, 1, 10_000)
    labels=np.random.uniform(0,1, len(arr))>-1
    bins = np.linspace(-5, 5, 11)
    weights = np.ones_like(arr)
    counts, _ = np.histogram(arr, bins=bins, weights=weights)
    
    #more advanced
    arr1 = np.random.normal(1, 1, size=10_000)
    labels1=np.random.uniform(0,1, len(arr1))>0.5
    arr2 = np.random.normal(-1, 1, size=5_000)
    labels2=np.random.uniform(0,1, len(arr2))>0.5
    
    def test_plot_stairs(self) -> None:
        figure_ax = plot.plot_stairs(self.counts, self.bins)
        assert isinstance(figure_ax, matplotlib.axes.Axes), "plot_stairs should ouput an axis!"

    def test_value_hist_uncertainty(self) -> None:
        weight_1 = plot.hist_uncertainty(self.arr, self.weights, self.bins)
        weight_2 = plot.hist_uncertainty(self.arr, self.weights*2, self.bins)
        assert all(weight_1 == weight_2/2), "Uncertainty not calculated correctly"

    def test_binned_errorbar(self) -> None:
        ax = plot.binned_errorbar(self.counts,
                             self.counts,
                             bins=self.bins)
        assert isinstance(ax, matplotlib.axes.Axes), "plot_stairs should ouput an axis!"
        
    def test_general_hist_plot(self):
        style = {"bins": self.bins, "histtype":"step"}
        counts_dict, ax = plot.plot_hist(self.arr, self.arr1, self.arr2,
                       weights=[self.weights, self.weights, self.weights[:len(self.arr2)]*2],
                       mask_sig=[self.labels, self.labels1, self.labels2],
                       style=style)
        assert isinstance(counts_dict, dict), "counts_dict not correct type"
        assert len(counts_dict)==4, "Not outputting the correct number of keys"
        n_sub_keys = np.sum([len(counts_dict[i]) for i in counts_dict])
        assert n_sub_keys==20, "Not outputting the correct number of sub keys"

        ax_2 = plot.plot_ratio(counts=counts_dict,
                               ax=ax, truth_key="dist_2",
                               ylim=[0.98,1.02])

if __name__ == "__main__":
    test = TestGeneralPlotting()
    counts_dict, something = test.test_general_hist_plot()