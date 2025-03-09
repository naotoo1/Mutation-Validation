# Mutation-Validation for Learning Vector Quantization
[Nana A. Otoo](https://github.com/naotoo1)


In this paper is posited a new model validation scheme for LVQ (Learning Vector Quantization) models. This repository contains all implementations of the mutation validation algorithm and models that were used for the evaluation. 

## Abstract
_Mutation validation as a complement to existing applied machine learning validation schemes has been explored in recent times. Exploratory work for Learning vector quantization (LVQ) based on this model validation scheme remains to be discovered. This paper proposes mutation validation as an extension to existing cross-validation and holdout schemes for Generalized LVQ and its advanced variants. The mutation validation scheme provides a responsive, interpretable, intuitive and easily comprehensible score that complements existing validation schemes employed in the performance evaluation of the prototype-based LVQ family of classification algorithms. This paper establishes a relation between the mutation validation scheme and the goodness of fit evaluation for four LVQ models: Generalized LVQ, Generalized Matrix LVQ, Generalized Tangent LVQ and Robust Soft LVQ models. Numerical analysis regarding model complexity and effects on test outcomes, pitches mutation validation scheme above cross-validation and holdout schemes_.

[https://vixra.org/abs/2308.0112](https://vixra.org/abs/2308.0112)


## About the models
The implementation requires Python 3.10 and above. The author recommends to use a virtual environment or Docker image. In  this regard, a fully reproducible environment using Nix and [devenv](https://devenv.sh/getting-started/) is highly recommended. Once you have installed Nix and devenv, you can do the following:

   ```bash
   mkdir -p ~/.config/nix
   echo "experimental-features = nix-command flakes" >> ~/.config/nix/nix.conf
   nix profile install --accept-flake-config "github:cachix/devenv/latest"
   ```

Then clone and enter the project directory:

```bash
git clone https://github.com/naotoo1/Mutation-Validation.git
cd Mutation-Validation
```

Activate the reproducible development environment:
   ```bash
devenv shell
   ```
You may optionally consider using [direnv](https://direnv.net/) for automatic shell activation when entering the project directory.

To install Mutation-Validation in a stable version, follow these steps to set up your environment with all the necessary dependencies with live code editing capabilities. To use the local reproducible environment, execute the following lock file commands:

```bash
# Generate requirements file
generate-requirements

# Update lock files
update-lock-files

# Install dependencies from lock file
install-from-lock
   ``` 

To use the reproducible docker container with support for GPU/CPU:

```bash
# For GPU support
create-reproducible-container
run-reproducible-container

# For CPU-only environments
create-cpu-container
run-cpu-container
   ```

To replicate an example training for Mutation Validation of WDBC dataset for Learning Vector Quantization as used in this paper run:
```python
python3 train.py
```

To replicate an example training for Mutation Validation of MNIST dataset for Learning Vector Quantization as used in this paper run:

```python
python3 train_1.py
```

The details of the implementation and results evaluation can be found in the paper. The results of MV, CV and Holdout schemes for two artificially generated datasets and two real-word datasets (WDBC and MNIST) against increasing LVQ model(s) complexity are presented in the paper. The target space perturbation algorithm presented in the paper is not only limited for use in LVQs but can also be used for any supervised machine learning/deep learning model. The mutation validation scheme presented in this paper can also be adopted in parallel to existing machine learning evaluation pipelines in runtime.  
