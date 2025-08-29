## Usage

### Dataset

In order to access and traverse through MERMAID data (including images and annotations) you would need to access the `mermaid_confirmed_annotations.parquet` file with a library such as `pandas`: 

```python
import pandas as pd 

annotations_path = "s3://coral-reef-training/mermaid/mermaid_confirmed_annotations.parquet" # Location of the annotations file

df_annotations = pd.read_parquet(annotations_path) # Pandas DataFrame containing each image alongside the 25 image annotations for benthic attributes and growth forms
    
df_images = df_annotations[ 
    ["image_id", "region_id", "region_name"]
].drop_duplicates(subset=["image_id"])  # Pandas DataFrame containing each image in the dataset
```

Then you can extract the image and annotations for a image from this DataFrame as such:

```python
import boto3
import io
from PIL import Image

def get_image_s3(
    image_id: str,
    s3: boto3.client,
    bucket: str = "coral-reef-training",
    thumbnail: bool = False,
):
    """
    Fetches an image from an S3 bucket and returns it as a PIL Image object.
    Args:
        image_id (str): The identifier of the image to retrieve.
        s3 (boto3.client): The boto3 S3 client used to access the bucket.
        bucket (str, optional): The name of the S3 bucket. Defaults to "coral-reef-training".
        thumbnail (bool, optional): If True, fetches the thumbnail version of the image. Defaults to False.
    Returns:
        PIL.Image.Image: The image loaded from S3.
    """
    key = (
        f"mermaid/{image_id}_thumbnail.png" if thumbnail else f"mermaid/{image_id}.png"
    )
    response = s3.get_object(Bucket=bucket, Key=key)
    image_data = response["Body"].read()

    image = Image.open(io.BytesIO(image_data))
    return image

s3 = boto3.client("s3") # Initialize S3 cliant using boto
idx = 0 # The index of the image we want to access - between 0 and the number of images 
image_id = df_images.loc[idx, "image_id"]
image = get_image_s3(image_id, s3, thumbnail=False).convert("RGB") # Retrieving the image from the bucket using the above-defined function

annotations = df_annotations.loc[df_annotations["image_id"] == image_id]
```

Alternatively, especially for machine learning use cases, you can use the code within the mermaidseg codebase, starting with the `MermaidDataset` class, which already includes this functionality and further steps allowing for ML research.

```python
from mermaidseg.datasets.dataset import MermaidDataset

dataset = MermaidDataset()

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