import torch
import torch.nn as nn
from torch.nn import functional as F
import mmap
import random
import pickle
import argparse

parser = argparse.ArgumentParser(description = "This is the demonstration program")

parser.add_argument('-batch_size', type = str, required = True, help = 'Please provide a batch_size')

args = parser.parse_args()

print(f'batch size: {args.batch_size}')

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(device)

block_size = 64
#batch_size = 128
batch_size = args.batch_size
max_iters = 3000
learning_rate = 3e-4
eval_iters = 100
eval_interval = 500
n_embd = 384
n_layer = 8
n_head = 8
dropout = 0.2 #drop neurons so we dont overfit, help model train, 

# chars = ""
# with open('wizard_of_oz.txt', 'r', encoding = 'utf-8') as f:
#     text = f.read()
#     chars = sorted(list(set(text)))
    

# print(chars)
# vocab_size = len(chars)

chars = ""
with open('vocab.txt', 'r', encoding = 'utf-8') as f:
    text = f.read()
    chars = sorted(list(set(text)))
    

print(chars)
vocab_size = len(chars)

string_to_int = { ch:i for i,ch in enumerate(chars) }
int_to_string = { i:ch for i,ch in enumerate(chars) }
encode =lambda s: [string_to_int.get(c, default_token) for c in s]
decode = lambda l: ''.join([int_to_string[i] for i in l])
default_token = len(chars) 
data = torch.tensor(encode(text), dtype = torch.long)
# # print(data[:100])

# n = int(0.8*len(data))
# train_data = data[:n]
# val_data = data[n:]

# def get_batch(split):
#     data = train_data if split == 'train' else val_data
#     ix = torch.randint(len(data) - block_size, (batch_size,))
#     x = torch.stack([data[i:i+block_size] for i in ix])
#     y = torch.stack([data[i+1:i+block_size + 1] for i in ix])
#     x, y = x.to(device), y.to(device)
#     return x,y

# x,y = get_batch('train')
# print('inputs: ')
# print(x)
# print('target: ')
# print(y)


def get_random_chunk(split):
    filename = "train.txt" if split == 'train' else "test.txt"
    with open(filename, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            # Determine the file size and a random position to start reading
            file_size = len(mm)
            start_pos = random.randint(0, (file_size) - block_size*batch_size)

            # Seek to the random position and read the block of text
            mm.seek(start_pos)
            block = mm.read(block_size*batch_size-1)

            # Decode the block to a string, ignoring any invalid byte sequences
            decoded_block = block.decode('utf-8', errors='ignore').replace('\r', '')
            
            # Train and test splits
            data = torch.tensor(encode(decoded_block), dtype=torch.long)
            
    return data


def get_batch(split):
    data = get_random_chunk(split)
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad() #not using grads to avoid additional computations and we are just doing evaluations
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



# Token Embeddings: Represent the individual meaning of each token (word or subword).
# Position Embeddings: Add information about where each token is located in the sequence (helps the model understand the order).
# Self-Attention: Allows each token to consider every other token in the sequence, adjusting its representation based on the relationships and context of other tokens.

#transformer block, the combination of self-attention, multi-head attention, and the feed-forward network (FFN) as its core components

#transformer architecture allows GPT to capture greater contextual meaning for each token by attending to all other tokens in the sequence (within the given block size), rather than just relying on adjacent tokens as simpler models like bigrams do.

# self-attention mechanism in transformers combines the dot product and softmax to compute weighted attention scores. Finally, these weights are used to form a weighted sum of the value vectors, capturing contextual relationships between tokens in the sequence.

class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.query = nn.Linear(n_embd, head_size, bias = False)
        self.value = nn.Linear(n_embd, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) #lower triangular matrix (trill) used to mask future positions in the attention mechanism 
        # ensures tokens can only attend to themselves and previous tokens.
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x) #(B, T, hs)
        q = self.query(x) #(B, T, hs)

        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 #Calculate attention scores using scaled dot product:
# q @ k.transpose(-2, -1): Computes the similarity between each token's query and all tokens' keys.
# k.shape[-1]**-0.5: Scales scores by the square root of head_size to stabilize gradients.
        #It tells the model how much "attention" each token should pay to every other token.
        # ensures that the attention weights are within a reasonable range for applying softmax, which converts them into probabilities for weighted token interactions.
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) #Mask future tokens by replacing their attention scores with -inf, ensuring that tokens cannot "see" forward in the sequence.
        wei = F.softmax(wei, dim = -1) #Convert attention scores into probabilities using the softmax function.
        wei = self.dropout(wei) # Randomly drop some attention weights to regularize the model

        v = self.value(x)
        out = wei @ v #Multiply attention weights (wei) with value vectors (v), combining information from attended tokens.
        return out #(B, T, head_size)
        
        









class MultiHeadAttention(nn.Module):
    #multiple heads of self-attention in parallel
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)]) #four heads running in parallel
        #nn.ModuleList is a container for a list of modules (e.g., individual attention heads) that PyTorch can recognize and manage.
        #[Head(head_size) for _ in range(num_heads)] creates num_heads instances of the Head class, each configured with head_size as the feature size.
        self.proj = nn.Linear(head_size * num_heads, n_embd) #project to an embd
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim = -1) #concatentate each of the heads along the feature dimension (B, T, [h1, h1, h1, h1, h2, h2, h2, h2] 4 features per head and 2 heads
        out = self.dropout(self.proj(out)) #self.proj(out) maps the concatenated output into a unified feature space of size n_embd and dropout randomly zeros out some features to prevent overfitting.
        return out




class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4* n_embd, n_embd),
            nn.Dropout(dropout), #prevent overfitting, drop neurons
        )
    def forward(self, x):
        return self.net(x)





class Block(nn.Module):

    def __init__(self, n_embd, n_head): #initalize transformation
        super().__init__()
        
        head_size = n_embd // n_head #number of features each head will capture
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd) #ReLU
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x): #post-normalization
        y = self.sa(x) #self attention
        x = self.ln1(x + y) #normalization
        y = self.ffwd(x) #feed Forward
        x = self.ln2(x + y) #normalization
        return x
        

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) #allows for simplicity in the later layers
        self.position_embedding_table = nn.Embedding(block_size, n_embd) #Using block size because we are considering the position in block. helps for transformers
        #The model uses block-based context for position embeddings
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)]) #creating a sequence of transformer blocks
        self.ln_f = nn.LayerNorm(n_embd) #applied to the input tensor to standardize it. This helps stabilize and accelerate the training process. 
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)
        
    def _init_weights(self, module): #apply initialization on weights
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean = 0.0, std=0.02)

    
    def forward(self, index, targets = None):
        B, T = index.shape
        

        tok_emb = self.token_embedding_table(index) # line looks up the token embeddings for the input indices
        pos_emb = self.position_embedding_table(torch.arange(T, device = device)) #generates the position embeddings for the input sequence
        x = tok_emb + pos_emb
        x = self.blocks(x) #applies each of the n_layer transformer blocks to the input x. Each transformer block contains self-attention and feed-forward layers that refine the embeddings.
        x = self.ln_f(x) #Layer normalization helps stabilize training and improve performance by standardizing the output from the transformer blocks
        logits = self.lm_head(x) #passes the normalized embeddings through the final linear layer (lm_head) to generate logits.
#Each layer builds upon the previous one, with the self-attention and feed-forward networks continually refining the context and meaning of each token

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape  #B batch size, T for Token Size, C for Vocab Size
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets) #compares the model's guess (logits) with the correct one (targets)
            
        return logits, loss
#The first logits are just the raw token embeddings, and
#The final logits are the model’s predictions for each token in the sequence after the embeddings have been refined by the model's layers.
    def generate(self, index, max_new_tokens):
        for _ in range(max_new_tokens):
            index_cond = index[:, -block_size:]
            logits, loss = self.forward(index_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim = -1)
            index_next = torch.multinomial(probs, num_samples = 1)
            index = torch.cat((index, index_next), dim = 1)
        return index


model = GPTLanguageModel(vocab_size)
# print('loading model parameter...')
# with open('model-01.pkl', 'rb') as f:
#     model = pickle.load(f)
# print('loaded successfully')
m = model.to(device)

# context = torch.zeros((1, 1), dtype = torch.long, device = device)
# generated_chars = decode(m.generate(context, max_new_tokens = 500)[0].tolist())
# print(generated_chars)


#pytorch optimizer

optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate) #This will help to adjust the model's parameters using gradients
#learning rate controls how big and small the adjustments are
for iter in range(max_iters):
    if iter % eval_iters == 0:
        losses = estimate_loss()
        print(f"step {iter}, train loss {losses['train']}, val loss: {losses['val']}")

    xb, yb = get_batch('train')

    logits, loss = model.forward(xb, yb)
    optimizer.zero_grad(set_to_none = True)
    loss.backward()
    optimizer.step() #Adjust the parameters using the gradients so that the next forward pass will make better predictions
print(loss.item())

with open('model-01.pkl', 'wb') as f:
    pickle.dump(model, f)

