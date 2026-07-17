# Facial Verification with a Siamese Neural Network

> One-shot face verification in TensorFlow/Keras. Two face images go in, a single similarity score comes out — telling you whether they belong to the same person.

This repository implements a **Siamese Neural Network** for facial verification. Instead of classifying a face into a fixed set of identities, the model learns an **embedding** for each image and compares two embeddings to decide whether they match. This makes it well suited to verification tasks where you only have a handful of reference images per person.

The project is implemented as a single Google Colab notebook (`model_engineering.ipynb`) and follows the classic Siamese-network approach popularized by [Nicholas Renotte's face-verification tutorial](https://www.youtube.com/watch?v=bK_k7eebGgc), itself inspired by the paper *["Siamese Neural Networks for One-shot Image Recognition"](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)* (Koch et al.).

---

## How It Works

A Siamese network passes two images — an **input/anchor** image and a **validation** image — through the *same* embedding network (shared weights). The two resulting embedding vectors are combined with a distance layer, and a final classifier turns that distance into a similarity score between 0 and 1.

```
input image  ─►┐
               ├─► [ shared embedding CNN ] ─► embedding_A ─┐
               │                                            ├─► L1 distance ─► Dense(sigmoid) ─► match / no-match
validation ───►┘                              embedding_B ─┘
```

- **Positive pair** (same person) → label `1`
- **Negative pair** (different person) → label `0`

At inference time, an input face is compared against one or more reference images; a score above the `0.5` threshold is treated as a match.

---

## Architecture

### Embedding network (`make_embedding`)

A convolutional feature extractor that maps a `100 x 100 x 3` image to a `4096`-dimensional embedding vector:

| Block | Layer | Details |
|-------|-------|---------|
| 1 | `Conv2D` | 64 filters, `10x10`, ReLU |
|   | `MaxPooling2D` | `2x2`, padding `same` |
| 2 | `Conv2D` | 128 filters, `7x7`, ReLU |
|   | `MaxPooling2D` | `2x2`, padding `same` |
| 3 | `Conv2D` | 128 filters, `4x4`, ReLU |
|   | `MaxPooling2D` | `2x2`, padding `same` |
| 4 | `Conv2D` | 256 filters, `4x4`, ReLU |
|   | `Flatten` | — |
|   | `Dense` | 4096 units, sigmoid → **embedding** |

### Distance layer (`L1Dist`)

A custom Keras `Layer` that computes the **element-wise L1 (absolute) distance** between the two embeddings:

```python
class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()

    def call(self, input_embedding, validation_embedding):
        input_embedding = tf.convert_to_tensor(input_embedding)
        validation_embedding = tf.convert_to_tensor(validation_embedding)
        return tf.math.abs(input_embedding - validation_embedding)
```

### Siamese model (`make_siamese_model`)

Two `100 x 100 x 3` inputs → shared embedding network → `L1Dist` → `Dense(1, activation='sigmoid')` classifier that outputs the match probability.

---

## Training

- **Loss:** Binary cross-entropy (`tf.losses.BinaryCrossentropy`) — the model is trained as a binary "same / different" classifier over image pairs.
- **Optimizer:** Adam with a learning rate of `1e-4`.
- **Custom training loop** using `tf.GradientTape` (`train_step`), wrapped with `@tf.function`.
- **Batch size:** 16, with `50` epochs in the notebook.
- **Checkpointing:** `tf.train.Checkpoint` saves the optimizer and model every 10 epochs to `./training_checkpoints`.
- **Evaluation:** `Precision` and `Recall` metrics from `tensorflow.keras.metrics`, evaluated on the test partition with a `0.5` decision threshold.

> Note: the metric values printed in the notebook are illustrative literals and should be re-measured on your own data.

---

## Dataset

The training data is organized into three folders of `250 x 250` JPEG images (300 sampled per class in the notebook):

```
data/
├── anchor/     # reference images of the target person
├── positive/   # more images of the same person (label 1 vs. anchor)
└── negative/   # images of other people (label 0 vs. anchor)
```

- **Anchor** and **positive** images are captured from a webcam (see `anchor_generator.py`).
- **Negative** images come from the [Labelled Faces in the Wild (LFW)](http://vis-www.cs.umass.edu/lfw/) dataset — the notebook contains a (commented-out) helper that flattens the LFW directory tree into the `negative/` folder.

Pairs are built as:

- `(anchor, positive, 1)`
- `(anchor, negative, 0)`

then concatenated, shuffled, and split **70% train / 30% test**.

### Preprocessing

Each image is read, decoded, resized to `100 x 100`, and scaled to the `[0, 1]` range.

---

## Tech Stack

- **Python 3**
- **TensorFlow / Keras** — model definition, custom training loop, metrics
- **OpenCV (`cv2`)** — webcam capture for anchor/positive images
- **NumPy** — array handling
- **Matplotlib** — visualizing image pairs and predictions
- **Google Colab** — the notebook mounts Google Drive and reads data from `drive/MyDrive/face_x/...`

> The repository does not pin exact library versions (no `requirements.txt`).

---

## How to Run

### 1. Collect your own anchor and positive images (optional)

`anchor_generator.py` opens your webcam and lets you capture cropped `250 x 250` frames:

```bash
python anchor_generator.py
```

- Press **`a`** to save an **anchor** image.
- Press **`p`** to save a **positive** image.
- Press **`q`** to quit.

Populate `negative/` with images from the LFW dataset.

### 2. Run the notebook

The notebook is written for **Google Colab**:

1. Open `model_engineering.ipynb` in Google Colab.
2. Mount Google Drive when prompted and place your data under `drive/MyDrive/face_x/data/{anchor,positive,negative}` (and `drive/MyDrive/face_x/lfw` for the raw LFW dataset).
3. Run the cells top to bottom to preprocess data, build the Siamese model, train it, and evaluate precision/recall.

To run locally instead, adjust the data paths (remove the Drive-mount and `drive/MyDrive/...` prefixes) and install the dependencies:

```bash
pip install tensorflow opencv-python numpy matplotlib
```

### 3. Save / reload the model

The trained model is saved to `siamesemodel.h5` and can be reloaded with the custom `L1Dist` layer:

```python
model = tf.keras.models.load_model(
    'siamesemodel.h5',
    custom_objects={'L1Dist': L1Dist,
                    'BinaryCrossentropy': tf.losses.BinaryCrossentropy}
)
```

---

## Project Structure

```
.
├── model_engineering.ipynb   # main notebook: data pipeline, model, training, evaluation
├── anchor_generator.py       # webcam capture script for anchor/positive images
└── README.md
```

---

## Credits

Architecture and workflow adapted from Nicholas Renotte's Siamese face-verification tutorial, based on Koch et al., *"Siamese Neural Networks for One-shot Image Recognition."*
