![alt text](https://github.com/DescoveryAmine/Senti-BERT/blob/main/multidialct_arabic_bert-SA.png?raw=true)
This is a repository of the Multi-dialect Sentiment Analysis BERT model.

By AmineDjebbi.

## About our Multi-dialect-Sentiment-BERT model:

This repository represents the first MultiDialect Attentive model Sentiment Analysis from BERT tuned from scratch and trained on 80k tweets of four different datasets (Tunisian, Egyptian, Algerian and Levant).

## Usage:

The model weights can be loaded using transformers library by HuggingFace.

model = AutoModel.from_pretrained("aminedjebbie/SentiBERT")


## Model Parameters:

| Parameter  | Value |
| ------------- | ------------- |
| architecture |	Uncased_L8_H12|
| hidden_size |	768 |
| max_position_embeddings  |	512 |
| num_attention_heads |	8 |
| num_hidden_layers |	12 |
| vocab_size |	51647 |
| Total number of parameters |	110M |

