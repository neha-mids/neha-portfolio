# W266-Final-Project
### Authors

* Sayan Das
* Neha Kumar

Descriptions of each of the files and folders are below:

best_model_output.txt: contains the text output of the best run on L3_classification. Visualizations and final findings are based on these results

W266_paper.pdf: Final Paper

Final Presentation_W266_Das_Kumar: Slides for final presentation

## data
metrics.xlsx: Contains results from all hyperparameter runs. Included in the appendix of the final paper

Visualizations.ipynb: Contains the code used to generate the graphics in the final paper

## data_collection_and_prep
L1 Data Prep.ipynb: Contains the code used to ingest the D1 (formerly known as L1) data from the Fake News Corpus in 10k row chunks and compile it into text files for pretraining

L2 Scraper.ipynb: Contains the code used to scrape websites on the 7 chosen political topics and combines the results into a single .txt file. Intermediate steps involved manually removing HTML formatting code that come through via the Scraper. This is the D2 data (formerly known as L2 Data)

Scraper_test_data.ipynb: Contains the code used to scrape information from the canddiates' interviews on the New York times as well as their websites / relevant platforms. Trump tweets were pulled from http://www.trumptwitterarchive.com/archive

L3 and Test Data Preprocessing.ipynb: Contains code to preprocess D3 (formerly known as L3 data) and Test data so it is ready for BERT and LSTM.

## training_scripts
L1_pretraining.ipynb: Runs the D1 (formerly known as L1) data using [Huggingface's run_language_modeling.py script](https://github.com/huggingface/transformers/blob/master/examples/run_language_modeling.py) to pre-train the model. This was split into 10 subtasks to be more manageable

L2_pretraining.ipynb: Runs the D2 (formerly known as L2) data using [Huggingface's run_language_modeling.py script](https://github.com/huggingface/transformers/blob/master/examples/run_language_modeling.py) to pre-train the model

L3_classification.ipynb: Runs the D3 (formerly known as L3) data and classification task using the L3_from_pretrain.py script

L3_from_pretrain.py: Performs the training job for the final Sequence classification task. Runs validation on the dev dataset and provides F1 scores. Provides extremity scores on the test dataset

lstm_baseline_cleaned.ipynb: Runs the baseline LSTM task

model_lstm.py: Contains Neural Network parameters for LSTM basline model

train_lstm.py: Training script for LSTM Baseline model

# References

## Coding Sources
* The library we used to implement BERT was the pytorch implementation found in Huggingface: https://github.com/huggingface/transformers
* For LSTM Baseline model (lstm_baseline_cleaned.ipynb, model_lstm.py, and train_lstm.py): https://github.com/danwild/sagemaker-sentiment-analysis
* Pretraining script was directly leveraged from [Huggingface](https://github.com/huggingface/transformers/blob/master/examples/run_language_modeling.py)
* For L3 and Test Data Preprocessing.ipynb: https://mccormickml.com/2019/07/22/BERT-fine-tuning/#3-tokenization--input-formatting
* For L3_from_pretrain.py:
    - https://github.com/huggingface/transformers/blob/master/examples/run_glue.py
    - https://aws.amazon.com/blogs/machine-learning/maximizing-nlp-model-performance-with-automatic-model-tuning-in-amazon-sagemaker/
    - https://github.com/danwild/sagemaker-sentiment-analysis/blob/163913a21837683e7605f6122ad2c10718347f65/train/train.py#L45
    - https://mccormickml.com/2019/07/22/BERT-fine-tuning/#3-tokenization--input-formatting


## Data Sources

### Test Data
- Websites that were scraped for Democratic candidates are within Scraper_test_data.ipynb
- Trump Tweets: http://www.trumptwitterarchive.com/archive

### D3 Data
D3 data sources are specfied as an additional column in the datasets [here](https://drive.google.com/drive/folders/1pTroDoyG3iIyQP7VcA2yVU3owkIPFjzM?usp=sharing), which were manually pulled

### D2 Data
Websites that were scraped for D2 data are within L2 Scraper.ipynb

### D1 Data
Pulled from the [Fake News Corpus](https://github.com/several27/FakeNewsCorpus/releases/tag/v1.0)
