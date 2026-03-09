"utils"
from tools.tools.visualization import general_plotting as plot
import io
from PIL import Image

def get_styles(best_path):
    lw=plot.lw
    if "mc_to_mc" in best_path.lower():
        style_target = {
            "marker": "o",
            "color": "black",
            "label": "Pythia",
            "linewidth": 0,
            "markersize": 4,
        }
        style_source = {
            "linestyle": "dotted",
            "color": "blue",
            "lw": lw,
            "label": "Herwig",
            # "drawstyle": "steps-mid",
        }
        style_trans = {
            "linestyle": "dashed",
            "color": "red",
            "lw": lw,
            "label": "Transport",
            # "drawstyle": "steps-mid",
        }
    else:
        style_target = {
            "marker": "o",
            "color": "black",
            "label": "Data",
            "linewidth": 0,
            "markersize": 4,
        }
        style_source = {
            "linestyle": "dotted",
            "color": "blue",
            "lw": lw,
            "label": r"$b$-jets",
            # "drawstyle": "steps-mid",
        }
        style_trans = {
            "linestyle": "dashed",
            "color": "red",
            "lw": lw,
            "label": "Transport",
            # "drawstyle": "steps-mid",
        }

    return style_source, style_trans, style_target



def jet_class_labels():
    return [
        r"p$_{QCD}$",
        r"p$_{Tbl}$",
        r"p$_{Tbqq}$",
        r"p$_{Wqq}$",
        r"p$_{Zqq}$",
        r"p$_{Hbb}$",
        r"p$_{Hcc}$",
        r"p$_{Hgg}$",
        r"p$_{H4q}$",
        r"p$_{Hqql}$",
        ]

def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img