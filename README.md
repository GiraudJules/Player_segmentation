# Player_segmentation
Football player segmentation project with images from the Kaggle dataset ["⚽ Football Player Segmentation ⚽"](https://www.kaggle.com/datasets/ihelon/football-player-segmentation) to use this model to create a Hockey or "all-sports" player segmentation dataset.

**⚠️ THIS PROJECT is STILL IN PROGRESS ⚠️**   
**Currently we can launch the code via notebooks, the implementation of .py files will follow.**

## Dataset

The dataset annotations are loaded from a JSON file, and images along with their corresponding masks are loaded into NumPy arrays. The dataset is then split into training, validation, and test sets. Images are resized for consistency.

## Model

The segmentation model is built using the `segmentation_models` library with a ResNet model as the backbone. Input images are preprocessed according to the requirements of the chosen backbone. The model is trained on the preprocessed training data.


## Installation Instructions

1. Clone the repository to your local machine.
```
git clone https://github.com/GiraudJules/Player_segmentation.git
```

2. Navigate to the project directory.
```
cd Player_segmentation
```

3. Create a virtual environment (optional but recommended).
```
python -m venv venv
source venv/bin/activate
# On Windows, use: venv\Scripts\activate
```

4. Install the required packages.
```
pip install -r requirements.txt
```

5. Launch Jupyter Notebook to run the provided notebooks.
```
jupyter notebook
```
## Dependencies

The project requires the following packages:

```
matplotlib
segmentation_models
Pillow
scikit-learn
opencv-python
tensorflow
numpy
imantics

```

## Usage

To replicate the results or build upon this project, start by running the `data_exploration.ipynb` notebook for data loading and visualization, followed by the `segmentation_model.ipynb` notebook for model training and evaluation.

---

For any additional details or queries, refer to the provided Jupyter notebooks.

