<div align="center">

# Calibrating latent spaces using Optimal Transport

[![python](https://img.shields.io/badge/-Python_3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![pytorch](https://img.shields.io/badge/-PyTorch_2.2-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![lightning](https://img.shields.io/badge/-Lightning_2.1-792EE5?logo=lightning&logoColor=white)](https://lightning.ai/)
[![hydra](https://img.shields.io/badge/-Hydra_1.3-89b8cd&logoColor=white)](https://hydra.cc/)
[![wandb](https://img.shields.io/badge/-WandB_0.16-orange?logo=weightsandbiases&logoColor=white)](https://wandb.ai)
</div>

This project is generated from the RODEM template for training deep learning models using PyTorch, Lightning, Hydra, and WandB. It is loosely based on the PyTorch Lightning Hydra template by [ashleve](https://github.com/ashleve/lightning-hydra-template).

## Submodules

This project relies on a custom submodule called `tools` stored [here](https://gitlab.cern.ch/malgren/tools) on CERN GitLab.
This is a collection of useful functions, layers and networks for deep learning developed by the RODEM group at UNIGE.

If you didn't clone the project with the `--recursive` flag you can pull the submodule using:

```
git submodule update --init --recursive
```

 - TODO also add OT framework

## Usage
### Train the transport
When the config have been set, its time to run the model:

```bash
python run/train_ot.py
```

### Configation files
The configuration files are manage by Hydra-core that is able to manage configuration settings for a machine learning project. Hydra is a powerful configuration management library that allows you to organize and override settings in a hierarchical and flexible manner. Let's break down the key components of this Hydra configuration file:
```yaml
hydra:
  run:
    dir: ${save_path}
```
* `hydra`: The root section for Hydra configuration.
* `run`: Configuration for the run mode.
* `dir`: The working directory for the run. It is set to ${save_path}, which is a variable that will be defined later in the configuration.

These are global hydra config settings


```yaml
defaults:
  - _self_
  - train: defaults.yaml
  - model: icnn.yaml
  - eval: defaults.yaml
  - data: defaults.yaml
```
* `defaults`: Specifies default configurations for different components of the project.
* `_self_`: Refers to the default configuration for the current file.
* `train`, `model`, `eval`, and `data`: These are references to other configuration files. When running the code, Hydra will merge the configurations from these files.

These are setting on how to train the model and config of the model.

## License
This project is licensed under the MIT License. See the [LICENSE](https://gitlab.cern.ch/rodem/projects/projecttemplate/blob/main/LICENSE) file for details.

