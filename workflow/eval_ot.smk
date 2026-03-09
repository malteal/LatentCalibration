# FILEPATH: /home/users/a/algren/work/diffusion/workflow/plot.smk
from tools import misc, smk_utils
from glob import glob

from snakemake.logging import logger

logger.info("your print statement here!")

configfile: "/home/users/a/algren/work/latn_calib/workflow/config/config_disc.yaml"
container: config["container_path"]
global_script_path = config['workdir']
global_save_path = config['global_path']
slurm_extra="--gres=gpu:1,VramPerGpu:2G --exclude=gpu002"

include: f"{global_script_path}/workflow/train_disc.smk"

wildcard_constraints:
    m_name = "[0-9a-zA-Z]+" # Makes sure the model name can't have underscores

# input files
all_inputs = []

# default configs and user defined variables
#  snakemake -s workflow/plot_template_generator.smk --workflow-profile workflow/profile_gpu/

eval_config = misc.load_yaml(f"{global_script_path}/configs/eval_ot.yaml")

# functions
convert_no_OT_calibs_to_bool = lambda x: 'no_OT' in x

def layers_to_proba(ot_path):
    if '128' in ot_path:
        return 8
    else:
        return 2

def path_to_latn(ot_path):
    if '128' in ot_path:
        return smk_utils.make_list_input_ready([8,2])
    else:
        return 'None'

# # input variables
# ot_paths = [
#     # f'{global_save_path}/JetClass/bb_latn_16_n_icnn_long_w_norm',
#     # f'{global_save_path}/JetClass/bb_latn_128_n_icnn_long_w_norm',
#     # f'{global_save_path}/JetClass/bb_latn_16_n_icnn_long_high_stats',
    
#     # 
#     f'{global_save_path}/JetClass/bb_latn_128_n_icnn_2',
#     # f'{global_save_path}/JetClass/bb_latn_16_n_icnn',
#     # f'{global_save_path}/JetClass/bb_128_only_model_n_icnn',
#             ]

# names
disc_physics = ['Tbqq_QCD']
# latn_size = ['16' if '16' in i else '128' for i in ot_path]
latn_sizes = ['16'] * len(ot_paths)
# bb_128_only_model_n_icnn do not have latn_16
if '128_only_model' in ot_paths:
    latn_sizes = None

all_roc_names = ['roc', 'roc_single_layer', 'roc_no_OT']

# get_1d_disc 
all_inputs += expand(
            [
            '{ot_path}/plots/source_all/latn_10/1d_disc/{disc_physics}.pdf',
            ], 
        disc_physics=disc_physics,
        ot_path=ot_paths,
        )

# get_probability_out
all_inputs += expand(
            [
            '{ot_path}/plots/source_all/latn_10/p_H4q.pdf',
            ], 
        ot_path=ot_paths,
        )

if latn_sizes is not None:
    # get_latn_out
    all_inputs += expand(
                [
                '{ot_path}/plots/source_all/latn_{latn_size}/latn0.pdf',
                ], 
            zip,
            ot_path=ot_paths,
            latn_size=latn_sizes
            )

    # get_corner
    all_inputs += expand(
                [
                '{ot_path}/plots/source_all/latn_{latn_size}/corner_latn_size_{latn_size}_target_v_transport.pdf',
                ], 
            ot_path=ot_paths,
            latn_size = latn_sizes
            )

# ROC of classifiers
all_inputs += expand(
            [
            '{ot_path}/plots/source_all/{all_roc_name}_and_auc.pdf',
            ], 
        ot_path=ot_paths,
        all_roc_name=all_roc_names
        )

def get_input_files(wildcards, clf_dims=clf_dims):
    if 'single' not in wildcards.all_roc_name:
        clf_models = ["classifier", "classifier_no_OT"]

    elif 'single' in wildcards.all_roc_name:
        clf_models = ["classifier_single_layer_no_OT", "classifier_single_layer"] 

    return [f"{wildcards.ot_path}/{model}_{dim}/plots_/_loss.png" 
            for model in clf_models for dim in clf_dims]

# remove double //
all_inputs = smk_utils.check_paths_for_warnings(all_inputs)

rule all:
    input: all_inputs

rule get_roc:
    input: 
        get_input_files
    output:
        '{ot_path}/plots/source_all/{all_roc_name}_and_auc.pdf',
    resources:
        gpu=1,
        slurm_extra=slurm_extra,
    params:
        use_calib = lambda wildcards: 'no_OT' in wildcards.all_roc_name,
        single_layer = lambda wildcards: 'single' in wildcards.all_roc_name,
    shell:
        """
        python /home/users/a/algren/work/latn_calib/evaluate/eval_roc_of_dics.py save_figures=True paths.JetClass.nominal={wildcards.ot_path} discriminator.use_calib={params.use_calib} discriminator.single_layer={params.single_layer}
        """

rule get_1d_disc:
    input: 
        '{ot_path}/transport_files/source_all.h5'
    output:
        '{ot_path}/plots/source_all/latn_10/1d_disc/{disc_physics}.pdf'
    params:
        layer_removed = lambda wildcards: layers_to_proba(wildcards.ot_path),
    resources:
        gpu=1,
        slurm_extra=slurm_extra,
    shell:
        """
        pip install corner --user &&
        python /home/users/a/algren/work/latn_calib/evaluate/eval_ot.py paths.JetClass.nominal={wildcards.ot_path} discriminators=True hist_kwargs=False run_clf_layers={params.layer_removed} save_figures=True
        """

rule get_probability_out:
    input: 
        '{ot_path}/transport_files/source_all.h5'
    output:
        '{ot_path}/plots/source_all/latn_10/p_H4q.pdf',
    params:
        layer_removed = lambda wildcards: layers_to_proba(wildcards.ot_path),
    resources:
        gpu=1,
        slurm_extra=slurm_extra,
    shell:
        """
        python /home/users/a/algren/work/latn_calib/evaluate/eval_ot.py paths.JetClass.nominal={wildcards.ot_path} discriminators=False hist_kwargs=True run_clf_layers={params.layer_removed} save_figures=True
        """

if latn_sizes is not None:
    rule get_latn_out:
        input: 
            '{ot_path}/transport_files/source_all.h5'
        output:
            '{ot_path}/plots/source_all/latn_{latn_size}/latn0.pdf',
        params:
            layer_removed = lambda wildcards: path_to_latn(wildcards.ot_path),
        resources:
            gpu=1,
            slurm_extra=slurm_extra,
        shell:
            """
            python /home/users/a/algren/work/latn_calib/evaluate/eval_ot.py paths.JetClass.nominal={wildcards.ot_path} discriminators=False hist_kwargs=True run_clf_layers={params.layer_removed} save_figures=True
            """

    rule get_corner:
        input: 
            '{ot_path}/transport_files/source_all.h5'
        output:
            '{ot_path}/plots/source_all/latn_{latn_size}/corner_latn_size_{latn_size}_target_v_transport.pdf',
        params:
            layer_removed = lambda wildcards: path_to_latn(wildcards.ot_path),
        resources:
            gpu=1,
            slurm_extra=slurm_extra,
        shell:
            """
            python /home/users/a/algren/work/latn_calib/evaluate/eval_ot.py paths.JetClass.nominal={wildcards.ot_path} discriminators=False hist_kwargs=False run_clf_layers={params.layer_removed} save_figures=True
            """

