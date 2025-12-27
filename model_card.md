# Model Card

## Model Details
This model is a supervised binary classification model trained to predict whether an individual earns more than $50K per year based on demographic and employment-related features from the U.S. Census dataset. The model is implemented using a RandomForestClassifier from the scikit-learn library. The training pipeline includes preprocessing of categorical features using one-hot encoding and label binarization.

## Intended Use
The model is intended for educational purposes to demonstrate the deployment of a scalable machine learning pipeline using FastAPI. It should not be used to make real-world financial, employment, or policy decisions. The predictions are probabilistic and depend heavily on the quality and representativeness of the training data.

## Training Data
The training data is derived from the publicly available U.S. Census Bureau dataset (Adult Census Income dataset). The dataset contains demographic information such as age, education, occupation, workclass, race, sex, and native country. The data was split into training and test sets using an 80/20 split.

## Evaluation Data
The evaluation data consists of the held-out 20% test split of the Census dataset that was not used during model training. This dataset follows the same schema and preprocessing steps as the training data.

## Metrics
The model performance was evaluated using precision, recall, and F1 score. These metrics were chosen to balance false positives and false negatives in a binary classification setting.

On the test dataset, the model achieved approximately:
- Precision: 0.73
- Recall: 0.62
- F1 Score: 0.67

Additionally, model performance was evaluated across slices of categorical features to assess fairness and robustness across subgroups.

## Ethical Considerations
The dataset includes sensitive demographic attributes such as race, sex, and native country. As a result, the model may exhibit bias against certain demographic groups. Care should be taken to avoid using this model in decision-making contexts that could negatively impact individuals or groups.

## Caveats and Recommendations
The model is trained on historical census data and may not generalize well to populations outside the United States or to more recent data. Future improvements could include hyperparameter tuning, bias mitigation techniques, and continuous monitoring of model performance in production.
