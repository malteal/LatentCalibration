# FILEPATH: /home/users/a/algren/work/diffusion/workflow/plot.smk
from snakemake.logging import logger

# input variables
ot_paths = (
   
    # used in paper
    f'{global_save_path}/JetClass/bb_128_only_model_n_icnn'

    # newer 2x classifier used in appendix
    # f'{global_save_path}/JetClass/bb_latn_128_n_icnn_2'
    # f'{global_save_path}/JetClass/bb_latn_128_n_icnn_3'
    # f'{global_save_path}/JetClass/bb_latn_128_n_icnn_4_big'

    # new GRL 
    # f'{global_save_path}/JetClass/bb_128_only_model_icnn_GRL_testing_lr'
)

logger.info(f"All OT paths: {ot_paths}")

rule predict_ot:
    output:
        '{ot_path}/transport_files/source_all.h5'
    resources:
        gpu=1,
        slurm_extra=slurm_extra,
    shell:
        """
        python /home/users/a/algren/work/latn_calib/run/predict_ot.py model_path={wildcards.ot_path} sample_name=source bkg_scaler_str=all
        """
