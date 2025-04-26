import torch
import torch.nn as nn
from torch.nn import functional as F

#hyperparameters
batch_size = 32 #how many independent sequences will we process in parallel?
block_size = 8 #what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 32
#---------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()


#here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
#create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars)}
itos = { i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

#Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+1+block_size] for i in ix])
    return x,y


#Average loss over a number of batches to get a less noisey measurement
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

#Memoryless transduction (Bigram model)
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        #this model just looks up the logits for the next token in a table indexed by the current token
        tok_emb = self.token_embedding_table(idx)
        pos_embed = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_embed
        logits = self.lm_head(x) 
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape #get the (B)atch size, number of (T)ime steps(context length), and number of (C)hannels (features)
            #Reshape to match torch's cross_entropy() specification
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits,targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        #idx is (B, T) array of indices 
        for _ in range(max_new_tokens):
            #get the predictions
            logits, loss = self(idx)
            #exctract the last timestep
            logits = logits[:,-1,:] #slice of shape (B,C)
            probs = F.softmax(logits, dim=-1) #get probs over the (C)hannels for each (B)atch
            #sample the distribution for each batch example
            idx_next = torch.multinomial(probs, num_samples=1) #tensor of shape (B, 1)
            #append idx_next to idx
            idx = torch.cat((idx, idx_next), dim=1) #(B, T+1)
        return idx
    
model = BigramLanguageModel(vocab_size)
m = model.to(device)

#create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

#Training loop
for iter in range(max_iters):

    #every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses= estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')

    #evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

#generate from the model
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))