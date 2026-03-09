from matplotlib import font_manager
import matplotlib.pyplot as plt

import logging
logging.basicConfig(level=logging.INFO)

try:
    from atlasify import atlasify
    import mplhep as hep
    font_dirs = ["/home/users/a/algren/work/old_projects/hep-mpl/"]
    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)

    for font_file in font_files:
        font_manager.fontManager.addfont(font_file)

    # setup styling
    hep.style.ATLAS["mathtext.sf"] = "TeX Gyre Heros:italic:bold"
    plt.style.use([hep.style.ATLAS])#, hep.style.firamath])
    plt.rc('font', family='Helvetica')
except ModuleNotFoundError:
    logging.warning("atlasify and mplhep in the environment")


############ PID labelling ############
pid_convertion = {
"part_isPhoton": [22],
"part_isNeutralHadron": [ ],
"part_isChargedHadron": [211, 321, 2212],
"part_isElectron": [11],
"part_isMuon": [13],
}

############ items to comment in or out depending on level in ATLAS ############
interal_str = 'Internal'
# interal_str = 'Preliminary'
# interal_str = ''

atlas_run2_string = r"$\sqrt{\mathrm{s}}=13$ TeV, 140 $\mathrm{fb}^{-1}$"
# atlas_run2_string = ""
############ items to comment in or out depending on level in ATLAS ############

### names ###
ftag_probs = [r"$p_b$", r"$p_c$", r"$p_u$"]
# ftag_probs_logit = [r"logit $\mathrm{p_b}$",
#             r"logit $\mathrm{p_c}$",
#             r"logit $\mathrm{p_u}$"]
ftag_probs_logit = [f"logit {i}" for i in ftag_probs]

dl1rb = r"$D_b^{\mathrm{DL1r}}$"
dl1rc = r"$D_c^{\mathrm{DL1r}}$"

pt_str = r"$p_\mathrm{T}$"
bjet = r"$b$-jet"
nonbjet = r"non-$b$-jet"

atlas_name = "$\\sf{ATLAS}$"

atlas_str =  f'{atlas_name} {interal_str}'
atlas_simulation_str =  f"{atlas_name} Simulation {interal_str}"


# plt.figure()
# plt.title(pt)

### ATLAS FTAG functions
def get_WP(value):
    if value==0.67:
        return "85% OP"
    elif value==2.20:
        return "77% OP"
    elif value==3.25:
        return "70% OP"
    elif value==4.57:
        return "60% OP"
    else:
        return f"cut at {value}"

operating_points = [0.67, 2.20, 3.25, 4.57]

### ATLAS style
def get_atlas_internal_str(lineskip=False, simulation=False) -> str:
    text = atlas_simulation_str if simulation else atlas_str
    if lineskip:
        text+="\n"
    return text

def get_atlas_legend(lineskip=False, pt:list=None):

    if pt is None:
        pt = [20, 400]

    # combining strings
    title=get_atlas_internal_str(True)
    data_sample=atlas_run2_string+"\n"
    pt_range = pt_str+r"$ \in $"+f"[{pt[0]}, {pt[1]}] GeV"
    string = title+data_sample+pt_range

    if lineskip:
        string += " \n"
    return string
    
def place_atlas_text(fig, ax, string, x_loc_for_legnd=None, ):
    
    if x_loc_for_legnd is None:
        # get where the legend is located
        transform = ax.transAxes.inverted()

        legend_bbox = ax.get_legend().get_window_extent(fig.canvas.get_renderer())
        bbox_figure = transform.transform(legend_bbox)
        x_loc_for_legnd = (0.2 if bbox_figure[0, 0]>0.35 else 0.8, 0.8)

    ax.text(*x_loc_for_legnd, string, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

def ATLAS_setup(ax,xlabel=None,ylabel=None, lumi_bool=True, 
                atlas_kw={"font_size": 16, "sub_font_size": 16, "label_font_size": 16,
                        #   "line_spacing":1.5
                          },
                text="",
                # legend_title='internal',
                simulation_text=True, **kwargs):

    # legend text
    if lumi_bool:
        text += atlas_run2_string+"\n"

    pt = kwargs.get("pt", [20, 400])
    if pt is not None:
        text += pt_str+r"$ \in $"+f"[{pt[0]}, {pt[1]}] GeV"
    
    eta = kwargs.get("eta")
    if eta is not None:
        text += f"\n$|\\it{{\\eta}}| \in [{eta[0]}, {eta[1]}]$"
        
    simulation_text = 'Simulation' if simulation_text else ''
    
    
    # if legend_title.lower() == 'internal':
    #     legend_title = f'{simulation_text} Internal'
    # elif legend_title.lower() == 'preliminary':
    legend_title = f"{simulation_text} {interal_str}"
    # else:
    #     legend_title = simulation_text+""
        
    if kwargs.get("lineskip", False):
        text += " \n"
    if kwargs.get("legend_kw") is not None:
        text= get_atlas_internal_str(True)+text
        ax.legend(title=text, **kwargs.get("legend_kw", {}))
    else:
        atlasify(legend_title, subtext=text, outside=False,
                 axes=ax, **atlas_kw
            # font_size=font, sub_font_size=font, label_font_size=font, line_spacing=1.5
        )
    if xlabel is not None:
        hep.atlas.set_xlabel(xlabel, ax=ax)

    if ylabel is not None:
        hep.atlas.set_ylabel(ylabel, ax=ax)

    return ax
