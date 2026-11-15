CAGEF is a PyTorch implementation for multi-omics classification when some samples lack an entire assay. It learns a representation for each observed modality, predicts the representations of missing modalities, and combines them to classify each sample. The experiment driver simulates assay-level missingness on complete numerical matrices and evaluates predictions on a held-out test set.
https://github.com/peng94673/CAGEF/blob/4b8f5e0f98165f45fb37065bc3e3aa03da700e99/CAGEF.png
Method
Encode each assay. A modality-specific network weights the input features and projects them into a common embedding space. Each assay can have a different number of input features.
Model relationships within a sample. The optional graph attention network treats the modalities of one sample as nodes. The availability mask identifies the assays that can contribute information.
Estimate missing representations. For each missing target modality, a predictor maps each observed source modality to the target embedding space. The model averages these predictions and estimates their confidence from the resulting embedding and the availability mask.
Fuse and classify. The model applies a learned attention weight to each modality embedding and a confidence weight to each estimated embedding. It concatenates the weighted embeddings and predicts class scores. Training combines cross-entropy with terms for embedding imputation, cross-modal consistency, and, when enabled, graph consistency.
Quick start
Use the implementation with the CAGEF.py entry point. If the source or data are distributed as archives, extract them first. Run the commands below from the directory containing CAGEF.py. Each example assumes the named dataset directory is in the working directory. For data stored elsewhere, pass its path to --data_folder.

CPU environment
The following package versions were used with Python 3.9.13 on Windows for the ROSMAP example. Run these commands in a clean Python environment.

python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy==1.24.4 scikit-learn==1.0.2 tqdm==4.64.1
Verify the installation
python CAGEF.py --data_folder ROSMAP --device cpu --num_runs 1 --num_folds 2 --num_epoch 1 --test_interval 1
This short run trains two models for one epoch each and writes predictions and summary metrics under exp/. It checks the data-loading, training, and evaluation workflow.

Run the default experiment
python CAGEF.py --data_folder BRCA
The default configuration runs five-fold cross-validation ten times, with a maximum of 2,000 epochs per fold. It selects CUDA when available and otherwise uses the CPU. Use a CUDA-enabled PyTorch installation for GPU training. Run python CAGEF.py --help for the full argument list.

Option	Default	Meaning
--modalities	All paired files	Select at least two modality prefixes, for example --modalities "1,3".
--missing_rate	0.1	Fraction of samples selected for simulated assay loss. Each selected sample retains at least one assay. Set to 0 for complete data.
--use_gnn	1	Set to 0 to disable the graph component.
--num_runs, --num_folds	10, 5	Number of repetitions and stratified folds per repetition.
--seed, --seed_step	58, 100	Initial seed and increment between repetitions.
--num_epoch, --test_interval, --patience	2000, 50, 15	Epoch limit per fold, validation interval in epochs, and early-stopping patience in validation checks.
--exp	./exp	Parent directory for timestamped experiment outputs.
Data
The four datasets used by this implementation each contain three modalities. The counts below refer to the supplied development and held-out test splits.

Dataset	Development samples	Test samples	Classes
BRCA	612	263	5
KIPAN	460	198	3
LGG	357	153	2
ROSMAP	245	106	2
BRCA uses the five PAM50 breast cancer subtypes. ROSMAP distinguishes Alzheimer's disease from normal controls. The dataset-specific README files for BRCA and ROSMAP document their label codes.

For each modality, provide a pair of headerless CSV files named <modality>_tr.csv and <modality>_te.csv. Rows represent samples and columns represent features. For example, 1_tr.csv and 1_te.csv contain the development and test matrices for modality 1. Use the same features in the same order within each pair.

Supply labels_tr.csv and labels_te.csv with one integer label per row. Within each split, all modality matrices and the label file must have identical sample order. Each development class must have at least as many samples as there are folds, and all test classes must occur in the development set. The loader requires finite numerical feature values. The experiment driver generates missing-modality masks after loading the data.

Evaluation and outputs
Within each repetition, the driver draws separate missing-modality masks for the development and test sets. All folds in that repetition share the same test mask. The held-out subjects stay fixed across repetitions, while the seed changes the fold assignments, simulated missingness, and model training.

Each fold fits feature scaling on its training partition. By default, validation ROC AUC determines checkpoint selection for binary tasks, and validation macro-F1 determines selection for multiclass tasks. Each selected checkpoint predicts the held-out samples. The driver averages the class probabilities across folds before scoring that repetition.

File	Contents
repeated_config.json	Experiment settings and the list of repetition seeds.
repeated_test_results.csv	Held-out ensemble scores for each repetition, followed by their mean and standard deviation. Binary tasks report accuracy, F1, and ROC AUC. Multiclass tasks report accuracy, weighted F1, and macro-F1.
repeated_summary.json	Machine-readable metrics across repetitions.
runs/<run>/summary.json	Fold-level validation metrics, pooled out-of-fold metrics, and held-out ensemble metrics for one repetition.
runs/<run>/predictions.npz	Out-of-fold and held-out class probabilities, encoded labels, fold assignments, modality names, and availability masks. The accompanying config.json records the label mapping.
runs/<run>/fold_<n>/checkpoint.pt	Model weights selected using that fold's validation data.
Source files
File	Responsibility
CAGEF.py	Parses experiment settings, schedules repetitions, and aggregates held-out results.
data.py	Reads paired modality matrices, generates availability masks, and augments observed inputs.
model.py	Defines modality encoders, graph attention, cross-modal predictors, confidence weighting, the classifier, and training losses.
trainer.py	Constructs folds, fits preprocessing, trains models, selects checkpoints, and evaluates predictions.
utils.py	Handles random seeds, device selection, tensor validation, and numerical operations.
