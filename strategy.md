# Machine Learning Classifiers

Our goal is to use machine learning techniques to classify input into one of three classes.

## Workflow

- Create a corpus where each entry corresponds to a patient and lists out the metabolites that are present in that patient.

- Perform GloVe embedding on that corpus and obtain vectors for each metabolite.

- Build out 220 feature vectors (1 for each patient). Each element within the feature vector is the concentration of the metabolite multiplied
with the vector, element-wise and then summed. Do this for each metabolite. There are 106 metabolites, so we should have 220 x 106 points.

- We can consider using the species, but there are thousands, so the dimensionality becomes far too large.

- Use KNN and Neural Network to classify based on this data.

## TODO

1. Create a new corpus.

2. Embed new corpus with GloVe.

3. Build out feature vectors.

4. Train and validate.