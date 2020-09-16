from transformers import TFAlbertForMaskedLM, TFAlbertModel, TFAlbertForSequenceClassification, AlbertForMaskedLM
import os

checkpoint = "albert-base-v1"

model = AlbertForMaskedLM.from_pretrained(checkpoint)

if not os.path.exists("~/saved/" + checkpoint):
    os.makedirs("~/saved/" + checkpoint)
    

model.save_pretrained("~/saved/" + checkpoint)
model = TFAlbertForMaskedLM.from_pretrained('~/saved/' + checkpoint, from_pt=True)
model.save_pretrained("~/saved/" + checkpoint)
model = TFAlbertModel.from_pretrained('~/saved/' + checkpoint)
model = TFAlbertForMaskedLM.from_pretrained('~/saved/' + checkpoint)
model = TFAlbertForSequenceClassification.from_pretrained('~/saved/' + checkpoint)


print("nice model") 