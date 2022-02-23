
# # Multi-dialect Sentiment analysis model for North African Arabic Language


import os
import datetime

from tqdm import tqdm

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras

import bert
from bert import BertModelLayer
from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights
from bert.tokenization.bert_tokenization import FullTokenizer



RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ## Import Data


def import_data(train,test,labels,samples):
    train = pd.read_csv(train)
    test = pd.read_csv(test)
    train.fillna(labels,inplace = True)
    test.fillna(samples,inplace = True)
    
    return train,test


train_set,test_set = import_data("Train_MultiCat.csv","Test_MultiCat.csv","polarite","message")


# ## Import Model


def import_model(dirt,path,chkpt,cfg):
    bert_model_name = dirt
    bert_ckpt_dir = os.path.join(path, bert_model_name)
    bert_ckpt_file = os.path.join(bert_ckpt_dir, chkpt)
    bert_config_file = os.path.join(bert_ckpt_dir, cfg)

    return bert_ckpt_dir,bert_ckpt_file,bert_config_file


from bert.tokenization.bert_tokenization import FullTokenizer


ckpt_dir,ckpt_file,config_file = import_model("uncased_L-8_H-768_A-12","model/","bert_model.ckpt","bert_config.json")


# ## Import Tokenizer

tokenizer = FullTokenizer(os.path.join(ckpt_dir, "Vocab_MultiCat.txt"))


# ## Prediction function

def predict_sentences(message):

 pred_tokens = map(tokenizer.tokenize, message)
 pred_tokens = map(lambda tok: ["[CLS]"] + tok + ["[SEP]"], pred_tokens)
 pred_token_ids = list(map(tokenizer.convert_tokens_to_ids, pred_tokens))

 pred_token_ids = map(lambda tids: tids +[0]*(data.max_seq_len-len(tids)),pred_token_ids)
 pred_token_ids = np.array(list(pred_token_ids))

 predictions = model.predict(pred_token_ids).argmax(axis=-1)

 for text, label in zip(message, predictions):
   print("text:", text, "\npolarity:", classes[label])
   print()


# ## Preprocessing


class IntentDetectionData:
  DATA_COLUMN = "message"
  LABEL_COLUMN = "polarite"

  def __init__(self, train, test, tokenizer: FullTokenizer, classes, max_seq_len=512):
    self.tokenizer = tokenizer
    self.max_seq_len = 0
    self.classes = classes
    
    train, test = map(lambda df: df.reindex(df[IntentDetectionData.DATA_COLUMN].str.len().sort_values().index), [train, test])
    
    ((self.train_x, self.train_y), (self.test_x, self.test_y)) = map(self._prepare, [train, test])

    print("max seq_len", self.max_seq_len)
    self.max_seq_len = min(self.max_seq_len, max_seq_len)
    self.train_x, self.test_x = map(self._pad, [self.train_x, self.test_x])

  def _prepare(self, df):
    x, y = [], []
    
    for _, row in tqdm(df.iterrows()):
      text, label = row[IntentDetectionData.DATA_COLUMN], row[IntentDetectionData.LABEL_COLUMN]
      tokens = self.tokenizer.tokenize(text)
      tokens = ["[CLS]"] + tokens + ["[SEP]"]
      token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
      self.max_seq_len = max(self.max_seq_len, len(token_ids))
      x.append(token_ids)
      y.append(self.classes.index(label))

    return np.array(x), np.array(y)

  def _pad(self, ids):
    x = []
    for input_ids in ids:
      input_ids = input_ids[:min(len(input_ids), self.max_seq_len - 2)]
      input_ids = input_ids + [0] * (self.max_seq_len - len(input_ids))
      x.append(np.array(input_ids))
    return np.array(x)


# ## Create Model

def create_model(max_seq_len, bert_ckpt_file):

  with tf.io.gfile.GFile(config_file, "r") as reader:
      bc = StockBertConfig.from_json_string(reader.read())
      bert_params = map_stock_config_to_params(bc)
      bert_params.adapter_size = None
      bert = BertModelLayer.from_params(bert_params, name="bert")
        
  input_ids = keras.layers.Input(shape=(max_seq_len, ), dtype='int32', name="input_ids")
  bert_output = bert(input_ids)

  print("bert shape", bert_output.shape)

  cls_out = keras.layers.Lambda(lambda seq: seq[:, 0, :])(bert_output)
  cls_out = keras.layers.Dropout(0.5)(cls_out)
  logits = keras.layers.Dense(units=768, activation="tanh")(cls_out)
  logits = keras.layers.Dropout(0.5)(logits)
  logits = keras.layers.Dense(units=len(classes), activation="softmax")(logits)

  model = keras.Model(inputs=input_ids, outputs=logits)
  model.build(input_shape=(None, max_seq_len))

  load_stock_weights(bert, bert_ckpt_file)
        
  return model


classes = train_set.polarite.unique().tolist()

data = IntentDetectionData(train_set, test_set, tokenizer, classes, max_seq_len=512)

model = create_model(data.max_seq_len, ckpt_file)

def main():


# ## Training



    model.compile(
      optimizer=keras.optimizers.Adam(1e-5),
      loss=keras.losses.SparseCategoricalCrossentropy(),
      metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")]
    )



    history = model.fit(
      x=data.train_x,
      y=data.train_y,
      validation_split=0.1,
      batch_size=4,
      shuffle=True,
      epochs=3
    )

# ## Evaluation

    predict_sentences(["بدي النيف عليك","دول الناس الغلط","ahsen rajel felalem","A7sen rajel fi tounes 😍"])


if __name__ == "__main__":

    main()



