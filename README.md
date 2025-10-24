# Amazon-ML-Challenge-2025
This README provides an overview of the approach used for the Amazon ML Challenge 2025 Competition, which involves multimodal price prediction leveraging text (BERT/SBERT), visual (DINOv2 image embeddings), and engineered numeric features to achieve a final ensemble SMAPE of 41.5%.

The core approach is a **multi-input regression model** where text features from different Transformer models are concatenated with features extracted from images (DINOv2 embeddings) and engineered numeric features, which are then passed to a regression head to predict the product price.

---

## 🚀 Project Overview: Multimodal Price Prediction

The goal of this project is to accurately predict the price of Amazon products using a combination of **Text**, **Image**, and **Numeric** data. We leverage advanced models like **BERT** and **Sentence-BERT (all-mpnet-base-v2)** for text, **DINOv2** for image embeddings, and a custom **k-Nearest Neighbors (kNN)** approach for contextual price features.

The final submission is an **ensemble** of the best-performing models, achieving a final validation **SMAPE of 41.5%**.

## 📁 File Structure and Components

The repository contains scripts for the full machine learning pipeline: data preprocessing, image embedding generation, model training, and ensembling.

### 1. Preprocessing and Feature Engineering

| File Name | Description |
| :--- | :--- |
| `create_train_val_splits.py` | Splits the original `train.csv` file into 85% training and 15% validation sets for model development. |
| `preprocess_stage1.py` | Extracts initial numeric **Value** and **Unit** information from the `catalog_content` field. It normalizes units (e.g., kg $\to$ gram, liter $\to$ ml, Ounces $\to$ oz) and uses **LLM fallback** for missing information. |
| `preprocess_stage2_final.py` | Takes the output of Stage 1 and converts all volume/weight units to a common base (`oz` and `fl_oz`). It also extracts **Pack Count** (`Count`) from product descriptions, resulting in three final engineered numeric features: `Count`, `oz`, and `fl_oz`. |

### 2. Image Download and Feature Extraction

| File Name | Description |
| :--- | :--- |
| `download_images.py` | A utility script that uses multithreading (`wget` via `subprocess`) to efficiently download images based on the provided links in the dataset. |
| `create_image_dino_embeddings.py` | Extracts image embeddings using the **DINOv2 (Vision Transformer Base)** self-supervised model. The script is configured for **Distributed Data Parallel (DDP)** processing, handles corrupted images with re-download logic, and saves the embeddings to a final `.pkl` file. |

### 3. Model Training

All core models are trained using **PyTorch DDP** for multi-GPU efficiency and **Mixed Precision (Autocast/GradScaler)** for faster training and reduced memory usage.

| File Name | Description |
| :--- | :--- |
| `BERT-DINO-trainer.ipynb` | The training notebook for the **BERT-based** multimodal model. The final model is a `TextPlusNumericModel` that concatenates BERT's `pooler_output` with the numeric features (`Count`, `oz`, `fl_oz`) and DINOv2 image embeddings. |
| `allmpnet-DINO-trainer.ipynb` | The training notebook for the **Sentence-BERT (all-mpnet-base-v2)** multimodal model. This model uses the SentenceTransformer's sentence embeddings instead of BERT's `pooler_output`. |
| `KNN_all_mp_net_trainer.py` | The training script for the specialized **kNN + all-mpnet-base-v2** model. This is an end-to-end model that integrates K-Nearest Neighbors predictions as additional features, using sentence embeddings to find neighbors in the training set and feeding their features/prices into the MLP head. |

### 4. Ensembling and Final Results

| File Name | Description |
| :--- | :--- |
| `ensembling-amazon-ml.ipynb` | The final script that orchestrates the prediction process for the trained BERT and SBERT models on the validation set. It then performs a **simple average ensemble** on their raw price predictions (`expm1(log(price+1))`) and calculates the final **SMAPE score**. |

---

## 📊 Performance Summary

The models were evaluated using the **Symmetric Mean Absolute Percentage Error (SMAPE)**, where a lower percentage is better.

| Model | Validation SMAPE | Notes |
| :--- | :--- | :--- |
| **BERT-DINO Model** | 43.65% | Strong performance, leveraging the full power of BERT for text. |
| **SBERT-DINO Model** | 44.85% | Good performance, using more efficient Sentence-BERT embeddings. |
| **kNN-SBERT Model** | 46.48% | Specialized contextual model, serving as a diverse ensemble member. |
| **Model Ensemble (Average)** | **41.5%** | **Best result achieved by averaging the BERT and SBERT predictions.** |

***Note:*** *Model weights files (`.pth`) have been excluded due to size limitations but are necessary to reproduce the prediction results.*
