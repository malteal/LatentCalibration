# FILEPATH: /home/users/a/algren/work/diffusion/workflow/plot.smk
from tools import misc, smk_utils
from glob import glob

configfile: "/home/users/a/algren/work/latn_calib/workflow/config/config_disc.yaml"
container: config["container_path"]
global_script_path = config['workdir']
global_save_path = config['global_path']

include: f"{global_script_path}/workflow/predict_ot.smk"

eval_config = misc.load_yaml(f"{global_script_path}/configs/eval_ot.yaml")

# functions
convert_no_OT_calibs_to_bool = lambda x: 'no_OT' in x

def get_run_clf_layers(ot_path):
    # should be updated with the one in train_dics
    if '128_only' in ot_path:
        run_clf_layers = {'null': None, 
                        #   '8-6': [8, 6], '8-4': [8, 4], 
                          '8-2': [8, 2], '8': 8}
    elif '128' in ot_path:
        run_clf_layers = {'null': None, '8-2': [8, 2], '8': 8}
    else:
        run_clf_layers = {'null': None, '2': 2}

    return run_clf_layers

run_clf_layers = get_run_clf_layers(ot_paths)
run_clf_layers = {k: smk_utils.make_list_input_ready(v) if isinstance(v, list) else v
                    for k, v in run_clf_layers.items()}

clf_dims = list(run_clf_layers.keys())
logger.info(f"Classifier dimensions: {clf_dims}")
logger.info(f"Layer to run: {run_clf_layers}")

# no_OT_calibs_strs = ['classifier', 'classifier_no_OT', 'classifier_single_layer_no_OT', 'classifier_single_layer']

rule train_disc:
    input:
        '{ot_path}/transport_files/source_all.h5'
    output:
        '{ot_path}/{no_OT_calibs_str}_{clf_dim}/plots_/_loss.png'
    params:
        single_layer = lambda wc: 'single_layer' in wc.no_OT_calibs_str,
        no_OT_calib = lambda wc: convert_no_OT_calibs_to_bool(wc.no_OT_calibs_str),
        layer_removed = lambda wc: run_clf_layers[wc.clf_dim],
    resources:
        gpu=1,
        slurm_extra=slurm_extra,
    shell:
        # pip install dotmap --user &&
        """
        python /home/users/a/algren/work/latn_calib/run/train_disc.py discriminator.no_OT_calib={params.no_OT_calib} run_clf_layers={params.layer_removed} paths.JetClass.nominal={wildcards.ot_path} discriminator.n_epochs=200 discriminator.single_layer={params.single_layer}
        """

