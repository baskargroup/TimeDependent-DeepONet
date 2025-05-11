# Benchmarking Scientific ML Models for Flow Prediction

This repository provides the source code and training scripts for **"Geometric Deep Operator Networks for Time-Dependent Prediction of Flows Over Varying Geometries"** The project evaluates the deep operator network **DeepONet** for predicting tranisent flow around complex geometries.

## Paper
Our study introduces a benchmark for scientific machine learning (SciML) models in predicting tranisnet flow over intricate geometries using high-fidelity simulation data. The full paper can be accessed here:

**"Geometric Deep Operator Networks for Time-Dependent Prediction of Flows Over Varying Geometries"** 
- Authors: *Ali Rabeh, Adarsh Krishnamurthy, Baskar Ganapathysubramanian*

## Datasets
This study utilizes the **FlowBench Flow Past Object (FPO) dataset**, which is publicly accessible on Hugging Face: [**FlowBench FPO Dataset**](https://huggingface.co/datasets/BGLab/FlowBench/tree/main/FPO_NS_2D_1024x256)

The dataset is licensed under **CC-BY-NC-4.0** and serves as a benchmark for the development and evaluation of scientific machine learning (SciML) models.

### Dataset Structure
- **Geometry representation:** SDF
- **Resolution:** 1024×256 (242 timesteps for each case)
- **Fields:** Velocity (u, v), Pressure (p)
- **Stored as:** Numpy tensors (`.npz` format)

# Installation  

## Essential Dependencies  

This repository requires the following core libraries: 

- **`torch`** – PyTorch framework for deep learning 
- **`pytorch-lightning`** – High-level PyTorch wrapper for training 
- **`omegaconf`** – Configuration management 
- **`wandb`** – Experiment tracking 
- **`numpy`** – Numerical computations 
- **`scipy`** – Scientific computing 

> **Note:** 
> We have included `venv_requirements.txt`, which lists all the libraries used in our environment. To set up the environment and install dependencies using `venv_requirements.txt`:
```bash
python3 -m venv sciml
source sciml/bin/activate 
pip install --upgrade pip setuptools wheel Cython
pip install -r venv_requirements.txt
```

## Model Training
To train the model, run the following command:
```bash
python3 main.py --config_path "path to conf.yaml"
```

Before training, you need to specify the dataset paths in the **configurations** (YAML files):
```yaml
data:
  file_path_train_x: ./data/train_x.npz
  file_path_train_y: ./data/train_y.npz
  file_path_test_x: ./data/test_x.npz
  file_path_test_y: ./data/test_y.npz
```

## Postprocessing
After training, evaluate erros and visualize results (for single_step and rollout) with:
```bash
python3 postprocess.py --checkpoint_path path/to/checkpoints/ --config_path path/to/conf.yaml --sample_ids "0,2,5"
```

## Contributing
We welcome contributions! Please fork the repository and submit a pull request.

## License
This repository is licensed under the MIT License.