from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained('gpt2')

generated = tokenizer.encode("The Manhattan bridge")
context = torch.tensor([generated])
past = None

for i in range(15):
    output, past = model(context, past=past)

    distribution = output[0, :]

    # Get the top 10 values' indices and cast them to a list
    top_values = distribution[-1].topk(10).indices.tolist()

    # Decode those into words
    top_words = [tokenizer.decode([x]) for x in top_values.indices.tolist()]

    # select words (only arbitrarily select the first three)
    words = words[0:3]

    # Cast them back to tokens which can be used as an added token
    selected_tokens = [tokenizer.encode(word) for word in words]

    generated += [argmax_token.tolist()]
    context = argmax_token.unsqueeze(0)

    print(tokenizer.decode([argmax_token.tolist()]))

sequence = tokenizer.decode(generated)

print(sequence)