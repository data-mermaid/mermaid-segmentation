# MermaidSeg

Codebase for the training, evaluation and usage of image segmentation models of coral data. 
The codebase contains data loaders, augmentations, preprocessors, models, training & evaluation scripts.

## File Structure

```
├── nbs/ # Jupyter notebooks
    ├── Base_Pipeline.ipynb # Notebook used to train end-to-end (non-concept) image segmentation models for a given dataset and configuration file
    ├── Concept_Bottleneck_Pipeline.ipynb # Notebook used to train end-to-end concept-bottleneck image segmentation models for a given dataset and configuration file
    ├── Model_Evaluation.ipynb # Notebook containing the code to quantitatively and qualitatively analyze the trained segmentation models 
    ├── datasets/
        ├── Dataset_Exploration.ipynb # Shows usage of currently implemented dataset classes and concept mapping
        ├── CoralNet_Annotations.ipynb # Notebook on how to map extracted CoralNet sources to a datasets readable format (not that relevant for now)
    ├── nb_experiments/
        ├── Time_Test.ipynb # Data I/O timing tests
        
├── mermaidseg/
    ├── datasets # Contains scripts related to dataset loading, preprocessing and data augmentations.
        ├── dataset.py # Contains dataset classes that can be used to acquire and load coral data, including images and annotations. Currently includes the MermaidDataset, CoralNetDataset and CoralscapesDataset classes.
        ├── concepts.py # Contains functionality related to working with concepts
        ├── utils.py # Contains utility functions related to datasets 
    ├── model # Contains everything related to training and evaluating segmentation models.
        ├── models.py # Contains the different model implementations
        ├── meta.py # Contains the metamodel class that is initialized with a model and set of hyperparameters, with train, eval and predict methods.
        ├── train.py # Contains the model training function.
        ├── eval.py # Contains evaluation classes & functions.
        ├── loss.py # Contains implementations of different loss functions.
    ├── io.py # Contains config & args set ups
    ├── logger.py # Contains classes & functions related to logging training with MLFlow and saving checkpoints & results
    └── visualization.py # Functions related to the visualization of images based on different criteria

├── configs/ # Configuration files for different runs/models
└── .gitignore 
└── .environment.yml # Environment for setup with conda/micromamba
└── .pyproject.toml
└── README.md
```

## Usage

### Installation

Prerequisites
- Python 3.11+
- AWS credentials configured with `mermaid-core` profile (for S3 access)
- Hugging Face account and token (for DINOv3 models; see [Hugging Face Setup](#hugging-face-setup) below)
- uv (recommended) or pip

**Quick setup with uv** (recommended):
```bash
git clone https://github.com/your-org/mermaid-segmentation.git
cd mermaid-segmentation

# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Configure AWS access
aws configure --profile mermaid-core
echo "AWS_PROFILE=mermaid-core" >> .env

# Configure Hugging Face (required for DINOv3 models)
hf auth login
# Or add your token to .env: echo "HF_TOKEN=hf_xxxxxxxxxxxx" >> .env

# Install dependencies
uv sync

```

**Traditional setup**:
```bash
git clone https://github.com/your-org/mermaid-segmentation.git
cd mermaid-segmentation
pip install -e .
# Then pip install any additional libraries required from the environment.yml file, depending on use case.
```


### Hugging Face Setup

DINOv3 models (e.g. `facebook/dinov3-vitb16-pretrain-lvd1689m`) are gated on Hugging Face and require authentication:

1. **Request access**: Visit the [model page](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m), sign in, and accept the license to request access.
2. **Authenticate** using one of these methods:
   - **CLI login** (recommended): Run `hf auth login` and paste your token when prompted.
   - **Environment variable**: Add `HF_TOKEN=hf_xxxxxxxxxxxx` to your `.env` file (create from `.env.example`). Create a token at [hf.co/settings/tokens](https://hf.co/settings/tokens).

If using `.env` with direnv, ensure you run `direnv allow` and restart the Jupyter kernel so it picks up the token.

### Dataset
In order to access and traverse through MERMAID data we can use the code within the mermaidseg codebase, starting with the `MermaidDataset` class, which already includes the functionality allowing for ML research.
Similarly, in order to access and traverse through CoralNet data we can use the analogous `CoralNetDataset` class that has the same functionality.

```python
from mermaidseg.datasets.dataset import MermaidDataset, CoralNet
from mermaidseg.datasets.utils import get_coralnet_sources

# Mermaid
dataset_mermaid = MermaidDataset()
idx = 0
image, mask, annotations = dataset[idx] # This function returns the image, annotations alongside a semantic segmentation mask.


# CoralNet
all_coralnet_sources = get_coralnet_sources()
dataset_coralnet = CoralNet(source_ids = all_coralnet_sources)
idx = 0
image, mask, annotations = dataset[idx] # This function returns the image, annotations alongside a semantic segmentation mask.
```

The image and annotations can be visualized based on their benthic attributes and growth forms as such:

```python
from matplotlib import pyplot as plt
from mermaidseg.visualization import get_legend_elements

fig, ax = plt.subplots(figsize = (8.5, 7), layout = "tight")
plt.imshow(image)
for i, annotation in annotations.iterrows():
    plt.scatter(annotation['col'], annotation['row'], 
                color=annotation["benthic_color"],
                marker=annotation["growth_form_marker"], 
                s=80,
                alpha=0.8)

benthic_legend_elements, growth_legend_elements = get_legend_elements(annotations)

first_legend = plt.legend(handles=benthic_legend_elements, bbox_to_anchor=(0.99, 1), 
                            loc='upper left', title='Benthic\nAttributes')
plt.gca().add_artist(first_legend)
plt.legend(handles=growth_legend_elements, bbox_to_anchor=(0.99, 0.4), 
          loc='center left', title='Growth\nForms')

plt.axis("off")
plt.show() 
```

### Segmentation Models
To train segmentation models using a specified config file one can use the `nbs/Base_Pipeline.ipynb` or `nbs/Concept_Bottleneck_Pipeline.ipynb` notebooks.

To evaluate any trained segmentation model, you can use the notebook `nbs/Model_Evaluation.ipynb` which contains both quantitative performance analyses through dataset level performance metrics and qualitative analyses by visualizing model results and corresponding class probabilities. 