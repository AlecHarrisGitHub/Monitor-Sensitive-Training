# MST News Article Bias POC

This folder contains a POC for Monitor Sensitive Training on biased news articles generated synthetically from various political perspectives. 

## Overview: 
* *Data Generation:* Approximately ~2300 diverse news topics under 40 different topic domains were generated using Claude 3.5 Sonnet. 1900 of those topics were used to generate short news articles with a randomly assigned bias profile (strong_left, center_left, center_right, strong_right). 
  * These topics and articles can be found at [kcorra716/bias-poc-data](https://huggingface.co/datasets/kcorra716/bias-poc-data). 
* *Training:* We then train two models: 
  * **Baseline**: Standard fine-tuning without bias labels. Each article is formatted alongside the instruction to "write a news article about [topic]" given a few contextual details. 
  * **MST**: Fine-tuning with bias category labels included in prompts. Each article is formatted alongside the instructions *and* the bias profile with which it was originally generated.
* *Evaluation:* 
  * Both models generate articles from the same topics. 
    * **Baseline**: Prompted without any monitor label (same as training)"
    * **MST**: Prompted with UNBIASED monitor label
  * Bias is evaluated using Claude 3.5 Sonnet with the following two metrics: 
    * Bias Score (0-10 scale, 0=unbiased, 10=extremely biased)
    * Objectivity Score (PASS, PARTIAL, and FAIL)
  * Key metrics: 
    * **Mean Bias Score**: Average bias across all articles
    * **MST Win Rate**: % of times MST has a lower bias than baseline on the same topic
    * **Improvement**: How much less biased MST is than baseline on the same topic

## Notebooks:
* `Bias_POC_Data_Generation.ipynb`: Generates the articles and topics used to train and evaluate the model. 
* `Bias_POC_Training.ipynb`: Performs LoRA fine-tuning with the synthetic data to produce the baseline and MST models. 
* `Bias_POC_Evaluation.ipynb`: Primitive evaluation of the models using new validation and test topics.

## Data & Models: 
* Datasets: [kcorra716/bias-poc-data](https://huggingface.co/datasets/kcorra716/bias-poc-data)
  * `all_articles.jsonl` - 1,859 training articles
  * `all_topics.jsonl` - 1,859 generated topics
  * `validation_topics.jsonl` - 200 validation topics
  * `test_topics.jsonl` - 200 test topics
* Models:
  * Baseline: [kcorra716/bias-poc-baseline](https://huggingface.co/kcorra716/bias-poc-baseline)
  * MST: [kcorra716/bias-poc-mst](https://huggingface.co/kcorra716/bias-poc-mst)

## How to Run: 
* *Required Keys:*
  * `OPENROUTER_KEY` - for Claude evaluator
  * `HUGGING_FACE_HUB_TOKEN` - for model downloads
* *Eval Start:*
  * `jupyter notebook notebooks/Bias_POC_Evaluation.ipynb`

## Progress Log:
**1/30: Waiting for more compute, code review before finishing evaluation & analysis**
