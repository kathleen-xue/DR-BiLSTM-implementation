from data import SNLI
import torch
from model import Bowman


device = torch.device('cuda')

snli = SNLI(batch_size=32, gpu=device)
model = Bowman(snli.TEXT.vocab)
model.load_state_dict(torch.load("./model/bowman_64.pth"))
model.to(device)

# first 5 premises with hypothesis
premises = ["This church choir sings to the masses as they sing joyous songs from the book at a church.",
"This church choir sings to the masses as they sing joyous songs from the book at a church.",
"This church choir sings to the masses as they sing joyous songs from the book at a church.",
"A woman with a green headscarf, blue shirt and a very big grin.",
"A woman with a green headscarf, blue shirt and a very big grin."]

hypothesis = ["The church has cracks in the ceiling.",
"The church is filled with song.",
"A choir singing at a baseball game.",
"The woman is young.",
"The woman is very happy."]

# ground truth
gold_label = ["neural", "entailment", "contradiction", "neural", "entailment"]

# tokenize
premises_token = [snli.TEXT.preprocess(x) for x in premises]
hypothesis_token = [snli.TEXT.preprocess(x) for x in hypothesis]

# label list
label_vocab = snli.LABEL.vocab.itos
preds = []

for i in range(len(premises)):
    # token to index in vocab
    prem, _ = snli.TEXT.numericalize(([premises_token[i]],[len(premises_token[i])]), device=device)
    hypo, _ = snli.TEXT.numericalize(([hypothesis_token[i]],[len(hypothesis_token[i])]), device=device)
    # do prediction
    output = model(prem, hypo)
    lab = label_vocab[int(torch.argmax(output))]
    preds.append(lab)

# print results
for i in range(len(premises)):
    print("Premise: " + premises[i])
    print("Hypothesis: " + hypothesis[i])
    print("Model Output: " + preds[i])
    print("Ground Truth: " + gold_label[i])
    print()