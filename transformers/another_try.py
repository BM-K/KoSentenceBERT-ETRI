from transformers import TFBertModel, BertTokenizer, BertConfig
import tensorflow as tf

config = BertConfig.from_pretrained("bert-base-cased", output_hidden_states=True)
model = TFBertModel.from_pretrained("bert-base-cased", config=config)

tok = BertTokenizer.from_pretrained("bert-base-cased")
text = tok.encode("Ain't this [MASK] best thing you've ever seen?")

inputs = tf.constant(text)
outputs = model.predict(inputs)

print(outputs)