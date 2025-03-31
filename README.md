## Advanced Machine Learning Project: Deep Learning for Raman Spectroscopy: A Study of CNNs and Vision Transformers for COVID-19 Classification

**Project Overview**

This project explores the application of deep learning techniques for classifying patients (COVID-19 positive, post-COVID negativized, and healthy controls) based on Raman spectroscopy data obtained from saliva samples. Raman spectroscopy is a non-invasive analytical technique that offers insights into the molecular composition of biological samples, making it promising for medical diagnostics, including the identification of biochemical alterations linked to COVID-19. The goal was to develop an automated diagnostic system and evaluate the effectiveness of various deep learning architectures, including both convolutional neural networks (CNNs) and Transformer-based models.

**Dataset**

The dataset used comprises 2400 Raman spectra acquired from saliva samples of 101 patients, divided into three categories:
* **COV+**: COVID-19 positive patients (30 patients, 720 spectra)
* **COV-**: Post-COVID negativized patients (37 patients, 888 spectra)
* **CTRL**: Healthy controls without prior infection (34 patients, 792 spectra)

Each spectrum contains Raman intensity as a function of Raman shift (cm⁻¹), along with the patient identifier and category label.

**Methodology**

1.  **Preprocessing:** To improve spectral data quality and prepare it for the models, several techniques were applied:
    * **Outlier Removal:** Identification and removal of saturated spectra based on an intensity threshold (60,000 a.u.) and repeated sequences. (87 spectra removed).
    * **Background Correction:** Application of a third-degree polynomial fitting to remove the background signal.
    * **Despiking:** Use of the Hayes-Whitaker algorithm based on modified Z-scores to detect and replace anomalous peaks (e.g., cosmic rays) with the average of neighboring values.
    * **Data Augmentation:** Generation of 7 synthetic spectra for each original spectrum by introducing random variations (offset, slope, multiplicative noise) to increase dataset size and variability.
    * **Normalization:** Standardization of data using `StandardScaler` from scikit-learn.
    * **Splitting:** Division of the dataset into training (70%), validation (15%), and test (15%) sets.

2.  **Modeling:** Various deep learning architectures for 1D data were implemented and evaluated:
    * **CNNs (TensorFlow/Keras):**
        * **VGG-like:** An architecture inspired by VGG, adapted for 1D data.
        * **ResNet:** A 1D implementation inspired by ResNet with residual blocks.
        * **DenseNet:** A 1D implementation inspired by DenseNet with dense blocks and concatenated connections.
        Hyperparameter optimization was performed for these architectures using KerasTuner (RandomSearch, 10 trials).
    * **Vision Transformer 1D (ViT1D - PyTorch):** A custom "from scratch" implementation of a Vision Transformer adapted to process 1D spectral data, exploring the use of self-attention mechanisms and patch embedding on this type of data.

**Technologies Used**

* **Language:** Python
* **Deep Learning:** TensorFlow, Keras, PyTorch
* **Machine Learning & Preprocessing:** Scikit-learn (`train_test_split`, `StandardScaler`, `LabelEncoder`, `classification_report`, `confusion_matrix`, `roc_auc_score`)
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn, Visual Keras
* **Hyperparameter Optimization:** KerasTuner
* **Other Libraries:** Pickle, SciPy, PIL, Torchinfo

**Results**

After the preprocessing phase and hyperparameter optimization, the CNN models showed high performance:

* **DenseNet:** Achieved the best overall accuracy on the test set at **87.4%**. It also showed balanced F1-scores across classes (COV+: 0.91, COV-: 0.86, CTRL: 0.85).
* **VGG-like:** Accuracy of **86.8%**.
* **ResNet:** Accuracy of **85.6%**.

The **Vision Transformer 1D (ViT1D)**, implemented as an exploratory model, demonstrated the feasibility of the Transformer-based approach for this data (test set accuracy around 67.6%), suggesting potential for future optimizations and research with larger datasets.
