# 📡 Intelligent Complaint Classification Using Localized Transformer Architecture

## Project Overview

The Nigerian telecommunications industry receives a large number of customer complaints about network problems, internet connectivity, billing, SIM services, fraud, and other telecom services.

Many complaints are written using **Nigerian English, Nigerian Pidgin, abbreviations, slang, misspellings, and local expressions**. This can make traditional keyword-based systems difficult to use effectively.

This project develops an **AI-powered complaint classification system** that uses a localized transformer architecture to understand and classify Nigerian telecom complaints.

The system uses **AfriBERTa Large** to classify complaints into eight specific intents and then maps each intent to an appropriate operational support team.

The project is designed to:

* Improve complaint classification
* Reduce manual triage
* Improve complaint routing
* Support faster responses
* Handle localized Nigerian communication
* Provide confidence scores
* Support human review for uncertain or sensitive complaints

The project focuses on using AI to **support customer-service teams rather than replace human decision-making**.

---

# 1. 🎯 Objectives

The main objectives of the project are to:

1. Automatically classify Nigerian telecom customer complaints.

2. Understand complaints written using Nigerian English, Nigerian Pidgin, slang, abbreviations, misspellings, and local expressions.

3. Reduce the amount of manual work required to categorize complaints.

4. Route complaints to the appropriate support teams.

5. Improve complaint-processing efficiency.

6. Provide confidence scores for model predictions.

7. Send uncertain complaints for human review.

8. Protect customer privacy during data preparation and processing.

9. Evaluate the model using reliable validation and test data.

10. Demonstrate how localized AI can improve customer-service systems in Nigeria.

---

# 2. 🛠️ Tools & Libraries

The project was developed using:

* **Python**
* **PyTorch**
* **Hugging Face Transformers**
* **AfriBERTa Large**
* **Scikit-learn**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Seaborn**
* **FastAPI**
* **Pydantic**
* **Kaggle Notebooks**
* **Git & GitHub**

### Core Model

```text
castorini/afriberta_large
```

AfriBERTa Large is used as the transformer backbone for understanding complaint text and predicting the complaint intent.

---

# 3. 📊 Dataset

The project uses two datasets.

### Training Dataset

The training dataset contains:

```text
9,029 complaints
4 columns
5 categories
8 intent classes
```

The main columns are:

| Column         | Description                 |
| -------------- | --------------------------- |
| `Complaint_id` | Unique complaint identifier |
| `Complaints`   | Customer complaint text     |
| `Category`     | Broad complaint category    |
| `Intent`       | Specific complaint intent   |

The model uses:

```text
Input  → Complaints
Target → Intent
```

### External Test Dataset

A separate external dataset contains:

```text
950 complaints
```

This dataset is used for production-style inference and complaint routing.

It is not used as the main accuracy benchmark because it does not contain the ground-truth labels required for evaluation.

### Dataset Sources

The Data Card describes the data as coming from **public sources supplemented with synthetic data**.

Public sources include:

* Social media platforms
* Telecommunications forums
* Digital review platforms

Synthetic data was also used to improve coverage of Nigerian Pidgin and mixed-language expressions.

---

# 4. 🏷️ Intent Classes

The model classifies complaints into eight intents.

| Intent                  | Meaning                                                           |
| ----------------------- | ----------------------------------------------------------------- |
| `abnormal_data_drain`   | Customer reports that data is being consumed unusually fast       |
| `forced_subscription`   | Customer reports an unwanted or unauthorized subscription         |
| `frequent_call_drops`   | Customer reports repeated call disconnections                     |
| `reported_scam`         | Customer reports a suspected scam or fraudulent activity          |
| `sim_barred_nin`        | Customer reports SIM/NIN-related barring or registration problems |
| `slow_internet`         | Customer reports slow or poor internet service                    |
| `total_outage`          | Customer reports complete loss of network/service                 |
| `unexplained_deduction` | Customer reports an unexplained deduction from their balance      |

These intents are used because they represent specific problems that can be connected to operational support teams. The Data Card describes the project as a multi-class classification system designed to map complaints to actionable support categories.

---

# 5. 🧹 Data Preprocessing

The complaint text is cleaned before it is given to the model.

The preprocessing includes:

* HTML entity cleaning
* URL removal
* HTML-like tag removal
* Social-media mention removal
* Nigerian phone-number masking
* Repeated punctuation reduction
* Repeated-character reduction
* Whitespace normalization

Phone numbers are replaced with:

```text
<PHONE_NUM>
```

The preprocessing does **not** remove Nigerian Pidgin, slang, or local expressions because these can contain useful information about the meaning of a complaint.

The process is:

```text
Raw Complaint
      ↓
Text Cleaning
      ↓
PII Masking
      ↓
Localized Language Preserved
      ↓
Cleaned Complaint
      ↓
Tokenization
```

The dataset was also checked for missing values and duplicates.

The notebook found:

```text
Missing complaint text: 0
Missing labels:         0
Duplicate complaint text: 0
```

The complaint-length analysis showed that most complaints are relatively short, with a 99th-percentile length of approximately 113 words.

A maximum sequence length of:

```text
128 tokens
```

was therefore used for the transformer.

---

# 6. 🤖 Modeling Approach

The project uses **AfriBERTa Large** for multi-class complaint classification.

The model is fine-tuned on the Nigerian telecom complaint dataset.

The training configuration includes:

| Parameter       |                        Value |
| --------------- | ---------------------------: |
| Model           |              AfriBERTa Large |
| Batch Size      |                            8 |
| Maximum Length  |                          128 |
| Epochs          |                            4 |
| Learning Rate   |                       `2e-5` |
| Weight Decay    |                       `0.01` |
| Dropout         |                        `0.2` |
| Dropout Samples |                            5 |
| Random Seed     |                           42 |
| Loss            | Class-weighted Cross-Entropy |

Because some intents contain more examples than others, inverse-frequency class weights are used during training.

This helps the model give appropriate attention to the less represented intent classes.

---

# 7. 🏗️ Architecture

The system uses a custom classification architecture built on top of AfriBERTa Large.

```text
Customer Complaint
        │
        ▼
Text Preprocessing
        │
        ▼
AfriBERTa Large
        │
        ▼
Token Embeddings
        │
        ▼
Attention-Masked Mean Pooling
        │
        ▼
5 Dropout Samples
        │
        ▼
Shared Linear Classifier
        │
        ▼
Averaged Logits
        │
        ▼
Predicted Intent
        │
        ▼
Confidence Score
```

### Attention-Masked Mean Pooling

The model calculates the average representation of the complaint while ignoring padding tokens.

This produces a single representation of the complaint that is passed to the classifier.

### Multi-Sample Dropout

Five dropout layers are used before the classification layer.

The resulting predictions are averaged to produce the final logits.

### Classification

The final layer predicts one of the eight complaint intents.

---

# 8. 📈 Evaluation

The dataset was divided using a **group-aware stratified split**.

The final split was:

| Dataset    | Samples | Percentage |
| ---------- | ------: | ---------: |
| Training   |   7,223 |        80% |
| Validation |     903 |        10% |
| Test       |     903 |        10% |

The notebook also performed a leakage check.

The result was:

```text
Train ↔ Validation: 0
Train ↔ Test:       0
Validation ↔ Test:  0
```

This confirms that the complaint texts did not overlap between the three partitions.

### Validation Performance

The best validation checkpoint was obtained at **Epoch 4**.

| Metric      |      Score |
| ----------- | ---------: |
| Accuracy    | **87.04%** |
| Macro F1    | **86.54%** |
| Weighted F1 | **87.09%** |

### Holdout Test Performance

The final model was evaluated on the separate 903-sample test split.

| Metric       |      Score |
| ------------ | ---------: |
| Accuracy     | **88.59%** |
| Macro F1     | **88.39%** |
| Weighted F1  | **88.57%** |
| Test Samples |    **903** |

The best-performing test intent by F1 was:

```text
slow_internet → 91.80%
```

The lowest-performing intent was:

```text
frequent_call_drops → 82.76%
```

The notebook also generated a confusion matrix and saved misclassified validation examples for further analysis.

---

# 9. 🚦 Intelligent Complaint Routing

The system does not stop after predicting an intent.

Each predicted intent is connected to an operational routing action.

| Intent                  | Category        | Routing Action                        |
| ----------------------- | --------------- | ------------------------------------- |
| `abnormal_data_drain`   | Mobile Data     | `ROUTE_TO_DATA_SERVICES_TEAM`         |
| `slow_internet`         | Mobile Data     | `ROUTE_TO_NETWORK_QUALITY_TEAM`       |
| `frequent_call_drops`   | Network Quality | `ROUTE_TO_VOICE_NETWORK_TEAM`         |
| `total_outage`          | Network Quality | `ESCALATE_TO_NETWORK_OPERATIONS_TEAM` |
| `unexplained_deduction` | Billing & VAS   | `ROUTE_TO_BILLING_TEAM`               |
| `forced_subscription`   | Billing & VAS   | `ROUTE_TO_VAS_SUPPORT_TEAM`           |
| `sim_barred_nin`        | SIM Services    | `ROUTE_TO_SIM_NIN_SUPPORT_TEAM`       |
| `reported_scam`         | Fraud           | `ESCALATE_TO_FRAUD_SECURITY_DESK`     |

The complete workflow is:

```text
Customer Complaint
        ↓
Intent Classification
        ↓
Confidence Assessment
        ↓
Category Identification
        ↓
Routing Decision
        ↓
Support Team
```

This supports the project's goal of reducing incorrect routing and improving complaint-processing efficiency. The Project Statement identifies faster routing and reduced manual workload as key goals of the system.

---

# 10. 🧑‍💼 Human Review

The system uses a confidence threshold of:

```text
0.85
```

If the model's confidence is below 0.85, the complaint is sent for human review.

```text
              Prediction
                   │
                   ▼
          Confidence ≥ 0.85?
             /          \
           Yes           No
            │             │
            ▼             ▼
       Auto Route     Human Review
```

There is also a special rule for:

```text
reported_scam
```

Fraud-related complaints are always escalated to:

```text
ESCALATE_TO_FRAUD_SECURITY_DESK
```

and require human review.

This is important because the project is designed to **support human decision-making rather than replace it**. The Project Statement specifically identifies transparency and accountability as principles requiring human involvement in uncertain or sensitive cases.

---

# 11. 🌐 API & Deployment

The trained model was integrated into a **FastAPI** application.

The API is called:


Intelligent Telecom Complaint Classification Engine


### Available Endpoints

GET /health
POST /classify


### Classification Request

The API accepts a complaint such as:


{
  "complaint_id": "TICK-101",
  "text": "MTN deducted 1500 naira from my balance without my permission for daily game service!"
}

The API returns:

Complaint ID
Cleaned Text
Predicted Intent
Predicted Category
Confidence
Routing Action
Human Review Status


### Example Result


Predicted Intent:
unexplained_deduction

Category:
Billing & VAS

Confidence:
0.9994

Routing:
ROUTE_TO_BILLING_TEAM

Human Review:
False


The API test returned:

HTTP 200


showing that the packaged model could successfully perform inference through the FastAPI application.

---

# 12. ⚖️ Responsible AI

Responsible AI is an important part of this project because the system influences how customer complaints are interpreted and routed.

The project follows six main principles.

### Fairness

Customers should receive the same quality of complaint handling regardless of whether they use formal English, Nigerian English, or Nigerian Pidgin.

The project therefore aims to recognize different Nigerian communication styles.

### Accessibility

The system is designed to recognize:

* Nigerian English
* Nigerian Pidgin
* Local expressions
* Abbreviations
* Informal language

Future versions should provide better support for customers who communicate mainly through indigenous Nigerian languages.

### Privacy

Publicly sourced complaint data was processed to remove personally identifiable information such as:

* Names
* Phone numbers
* Banking information
* Account information

The notebook also masks detected Nigerian phone numbers during preprocessing.

### Transparency

Users should know when automated systems are being used to classify their complaints.

There should also be a clear way to challenge or correct an incorrect classification. The Stakeholder Engagement Plan recommends plain-language communication and an easy path for customers to flag incorrect routing.

### Accountability

The system supports human decision-making.

Humans remain responsible for important decisions, especially for uncertain, ambiguous, or sensitive complaints.

### Reliability

The model should be continuously evaluated across different complaint categories and language styles.

Regular evaluation, error analysis, and feedback are required before wider deployment.

---

# 13.  Limitations

The current system has several limitations.

### 1. Dataset Representation

The dataset does not represent every Nigerian telecom customer.

People who complain through offline channels, voice calls, or other non-digital methods may be underrepresented. Rural populations and users with limited internet access may also be underrepresented.

### 2. Indigenous Languages

The project focuses mainly on Nigerian English, Pidgin, and localized expressions.

It does not yet provide complete coverage of all indigenous Nigerian languages.

### 3. AI-Generated Labels

The Data Card identifies the dataset labels as AI-generated/auto-assigned.

Human review is therefore needed to check label consistency and accuracy, especially for Nigerian Pidgin and local dialects.

### 4. Single-Intent Classification

A real customer may report more than one problem in the same complaint.

The current model predicts one primary intent.

### 5. Model Confidence

A high confidence score does not guarantee that the prediction is correct.

For this reason, confidence is used as a triage signal rather than a guarantee of correctness.

### 6. Prototype Routing

The routing actions demonstrated in this project are prototype operational mappings.

A real deployment would require integration with actual telecom customer-service, billing, fraud, and network-management systems.

---

# 14. 📁 Repository Structure

The current repository contains:


C10_Team_Gongola_NEW/
│
├── README.md
│
├── data/
│   └── README.md
│
├── docs/
│   ├── Data Card.pdf
│   ├── Impact Statement.pdf
│   ├── Problem Statement.pdf
│   └── Stakeholder Engagement Plan.pdf
│
└── scripts/
    ├── README.md
    └── intelligent-complaint-classifier.ipynb


The main notebook contains the complete machine-learning workflow:

```text
Dataset Loading
      ↓
Data Analysis
      ↓
Data Cleaning
      ↓
Label Mapping
      ↓
Data Splitting
      ↓
Leakage Checking
      ↓
Class Weighting
      ↓
AfriBERTa Training
      ↓
Validation
      ↓
Error Analysis
      ↓
Model Packaging
      ↓
Test Evaluation
      ↓
Production Inference
      ↓
FastAPI Deployment

How to Clone the Repository

A visitor can download the project to their computer by following these steps:

Step 1: Install Git

Download and install Git if it is not already installed on your computer.

Step 2: Open a terminal

Open Command Prompt, PowerShell, Git Bash, or the terminal in VS Code.

Step 3: Clone the repository

Run:

git clone https://github.com/chukwubuikemokoli963-max/C10_team_gongola.git

Step 4: Enter the project folder

cd C10_Team_Gongola_NEW

Step 5: Open the project

If you use VS Code:

code .

You can then open:

scripts/intelligent-complaint-classifier.ipynb

to view the notebook.

Alternative: Download Without Git

Visitors who do not have Git can also:

Open the repository on GitHub.
Click the Code button.
Select Download ZIP.
Extract the ZIP file.
Open the extracted C10_team_gongola folder.

---

# 15. 👥 Contributors

### |Team Mentor
**Mr David Balogun**

### Team Lead

**Chukwubuikem Okoli**

### Team Members

* **Uche Jumbo** 
* **Melchizedek Madaki**
* **Shedrach Okute** 

### Cohort

**TRI AI Saturdays — Cohort 10**

---

# 16. 📚 Documentation

The project includes four main documents:

### Problem Statement

Explains the customer-service problem, why existing systems struggle with Nigerian communication styles, and the proposed AI solution.

### Data Card

Documents the dataset, data sources, labeling, preprocessing, privacy considerations, representation, and intended use.

### Impact Statement

Describes the expected benefits, potential risks, affected communities, and mitigation strategies.

### Stakeholder Engagement Plan

Identifies the main stakeholders:

* Customers / Complainants
* Business / Support Leadership

It also describes how customers should be kept informed and how business/support leadership should be involved in deployment decisions.

---

# 17. 🙏 Acknowledgment

This project was developed as part of **TRI AI Saturdays Cohort 10**.

We acknowledge the TRI AI community, facilitators, mentors, and fellow participants for providing the learning environment and support that contributed to the development of this project.

The project was developed with the goal of demonstrating how responsible AI can be applied to a practical problem affecting customer-service delivery in Nigeria.

---

# 18. 📖 References

### Model

**AfriBERTa Large**

castorini/afriberta_large


### Dataset

**Nigerian Telecom Complaint Classification Dataset**

The project uses the training and external test datasets used in the Kaggle notebook.

### Responsible AI

* NIST AI Risk Management Framework
* OECD AI Principles
* Nigeria Data Protection Commission
* Nigerian Communications Commission

### Project Documentation

* `Problem Statement.pdf`
* `Data Card.pdf`
* `Impact Statement.pdf`
* `Stakeholder Engagement Plan.pdf`

---

# 🚀 Project Summary

The project provides an AI-assisted approach to Nigerian telecom complaint handling.


Customer Complaint
        ↓
Text Preprocessing
        ↓
AfriBERTa Large
        ↓
Intent Classification
        ↓
Confidence Assessment
        ↓
Human Review / Automatic Routing
        ↓
Support Team


The model achieved **88.59% accuracy and 88.39% Macro F1** on the 903-sample held-out benchmark test set.

The project shows how localized Natural Language Processing can be used to improve complaint classification while considering **fairness, accessibility, privacy, transparency, accountability, and reliability**.

The goal is not to remove humans from customer support.

The goal is to help support teams **understand complaints faster, route them correctly, and provide better service to customers using different Nigerian communication styles**.
