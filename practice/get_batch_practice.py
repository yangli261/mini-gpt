import torch
from torch import nn
from torch.nn import functional as F
import math

text = "banana"
chars = sorted(set(text))
stoi = {char:index for index,char in enumerate(chars)}
itos = {index:char for index,char in enumerate(chars)}

def encode(s):
    return [stoi[char] for char in s]

def decode(ids):
    return "".join(itos[id] for id in ids)


data = torch.tensor(encode(text),dtype = torch.long)
T = 3
valid_starts = list(range(len(data) - T))
x = torch.stack([data[start:start + T] for start in valid_starts])
y = torch.stack([data[start + 1:start + T + 1] for start in valid_starts]) 

assert x.shape == y.shape == (3,3),f"实际形状为x：{x.shape},y:{y.shape}"

first_pairs = (decode(x[0].tolist()),decode(y[0].tolist()))
assert first_pairs == ("ban","ana"),f"实际第一对为{first_pairs}"
print(f"第一对为{first_pairs}")

device = "cuda"

class BiagramModel(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.table = nn.Embedding(vocab_size,vocab_size)
        nn.init.zeros_(self.table.weight)
    def forward(self,idx):
        return self.table(idx)

model = BiagramModel(T).to(device)
x,y = x.to(device),y.to(device)
assert model(x).shape == (3,3,3),f"实际形状为：{model(x).shape}"
loss = F.cross_entropy(
    model(x).reshape(-1,T),
    y.reshape(-1)
)
assert math.isclose(loss.item(),math.log(3),rel_tol = 1e-6),f"实际损失为{loss.item()}"
model.eval()
with torch.no_grad():
    tokens = torch.tensor(
        [encode("b")],
        dtype = torch.long,
        device = device
    )
    for _ in range(5):
        logits = model(tokens[:,-1:])[:,-1,:]
        probs = logits.softmax(dim = -1)
        next_id = torch.multinomial(probs,num_samples = 1)
        tokens = torch.cat((tokens,next_id),dim = 1)
    assert tokens.shape == (1,6),f"实际形状为：{tokens.shape}"
    samples = decode(tokens[0].tolist())
    print(f"生成样本为：{samples}")

