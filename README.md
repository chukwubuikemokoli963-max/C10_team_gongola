# C10_Team_Gongola

## Intelligent Complaint Classification Using Localized Transformer Architecture 

TRI AI Saturdays Cohort 10 project for classifying Nigerian telecommunications customer complaints into complaint intents and routing actions.

## Project Structure


C10_Team_Gongola/
├── README.md
├── docs/
│   ├── Problem Statement.pdf
│   ├── Data Card.pdf
│   ├── Impact Statement.pdf
│   └── Stakeholder Engagement Plan.pdf
├── scripts/
│   ├── intelligent-complaint-classifier.ipynb
│   └── README.md
└── data/
    └── README.md


## Dataset

The final notebook loads the competition train and test CSV files from Kaggle. 

The model predicts the `Intent` label. The eight intents are:
`abnormal_data_drain`, `slow_internet`, `frequent_call_drops`, `sim_barred_nin`, `unexplained_deduction`, `total_outage`, `reported_scam`, and `forced_subscription`.

## Training Pipeline

The notebook performs dataset ingestion, class-distribution analysis, missingness and duplicate audits, complaint-length analysis, label mapping, text preprocessing, empty-record validation, group-aware stratified splitting, leakage checks, inverse-frequency class weighting, AfriBERTa training, error analysis, benchmark evaluation, checkpoint packaging, and production-style inference.

## Model

The notebook fine-tunes `castorini/afriberta_large` using attention-masked mean pooling, five parallel dropout layers with averaged classification outputs, and a linear classification head.

Key settings in the notebook:
- Seed: 42
- Maximum sequence length: 128
- Batch size: 8
- Epochs: 4
- Learning rate: 2e-5
- Weight decay: 0.01
- Early-stopping patience: 2

## Reproduction

The project code snippet is the notebook in `scripts/`.

Run the notebook in Kaggle with the same  datasets/input as provided in the data/ folder.


## Documentation

The `docs/` directory contains the Problem Statement, Data Card, Impact Statement, and Stakeholder Engagement Plan.

## Cohort: TRI AI Saturdays Cohort10

## Contributors

- Mr David Balogun Team mentor 
- Chukwubuikem Okoli — Team Lead
- Uche Jumbo
- Melchizedek Madaki
- Shedrach Okute



