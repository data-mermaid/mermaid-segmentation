# MermaidSeg

Codebase for the training, evaluation and usage of image segmentation models of coral data. 
The codebase contains data loaders, augmentations, preprocessors, models, training & evaluation scripts.

## File Structure

```
├── nbs/ # Jupyter notebooks
    ├── Model_Run.ipynb # Notebook used to train end-to-end image segmentation models for a given dataset and configuration file
    ├── Model_Evaluation.ipynb # Notebook containing the code to quantitatively and qualitatively analyze the trained segmentation models 
├── mermaidseg/
    ├── datasets # Contains scripts related to dataset loading, preprocessing and data augmentations.
        ├── dataset.py # Contains dataset classes that can be used to acquire and load coral data, including images and annotations. Currently includes the MermaidDataset and CoralNetDataset classes.
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
- AWS credentials configured (to from S3)

Install locally:
```bash
git clone https://github.com/your-org/mermaid-segmentation.git
cd mermaid-segmentation
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -e .
```

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
To train segmentation models using a specified config file one can use the `nbs/Model_Run.ipynb` notebook or run the corresponding script using ```python scripts/train.py`.

To evaluate any trained segmentation model, you can use the notebook `nbs/Model_Evaluation.ipynb` which contains both quantitative performance analyses through dataset level performance metrics and qualitative analyses by visualizing model results and corresponding class probabilities. 