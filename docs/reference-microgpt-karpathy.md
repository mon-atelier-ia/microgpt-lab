# Référence — microgpt.py & guide Karpathy

> **Contexte LLM** : Ce document est la source de vérité pour le projet microgpt-visualizer-fr.
> Il capture le contenu intégral du code source (`microgpt.py`) et du guide pédagogique de Karpathy
> afin d'éviter des fetches web répétitifs lors des sessions de travail.
>
> **Sources web** :
>
> - Guide : <https://karpathy.github.io/2026/02/12/microgpt/>
> - Code source (gist) : <https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95>
> - Progression (build_microgpt.py) : <https://gist.github.com/karpathy/561ac2de12a47cc06a23691e1be9543a>
> - Dataset (names.txt) : <https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt>
>
> **Date de capture** : 2026-03-01
> **Auteur original** : Andrej Karpathy (@karpathy)
> **Relation avec l'app** : `src/engine/model.ts` est un port TypeScript exact de `microgpt.py`.
> Les pages de l'app visualisent les concepts décrits dans le guide.

---

## Table des matières

1. [Code source complet (microgpt.py)](#1-code-source-complet)
2. [Guide — Dataset](#2-dataset)
3. [Guide — Tokenizer](#3-tokenizer)
4. [Guide — Autograd](#4-autograd)
5. [Guide — Parameters](#5-parameters)
6. [Guide — Architecture](#6-architecture)
7. [Guide — Training loop](#7-training-loop)
8. [Guide — Inference](#8-inference)
9. [Guide — Progression](#9-progression)
10. [Guide — Real stuff (scaling)](#10-real-stuff)
11. [Guide — FAQ](#11-faq)
12. [Correspondance guide ↔ app](#12-correspondance)

---

## 1. Code source complet

```python
"""
The most atomic way to train and run inference for a GPT in pure, dependency-free Python.
This file is the complete algorithm.
Everything else is just efficiency.

@karpathy
"""

import os       # os.path.exists
import math     # math.log, math.exp
import random   # random.seed, random.choices, random.gauss, random.shuffle
random.seed(42) # Let there be order among chaos

# Let there be a Dataset `docs`: list[str] of documents (e.g. a list of names)
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')
docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")

# Let there be a Tokenizer to translate strings to sequences of integers ("tokens") and back
uchars = sorted(set(''.join(docs))) # unique characters in the dataset become token ids 0..n-1
BOS = len(uchars) # token id for a special Beginning of Sequence (BOS) token
vocab_size = len(uchars) + 1 # total number of unique tokens, +1 is for BOS
print(f"vocab size: {vocab_size}")

# Let there be Autograd to recursively apply the chain rule through a computation graph
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads') # Python optimization for memory usage

    def __init__(self, data, children=(), local_grads=()):
        self.data = data                # scalar value of this node calculated during forward pass
        self.grad = 0                   # derivative of the loss w.r.t. this node, calculated in backward pass
        self._children = children       # children of this node in the computation graph
        self._local_grads = local_grads # local derivative of this node w.r.t. its children

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other): return Value(self.data**other, (self,), (other * self.data**(other-1),))
    def log(self): return Value(math.log(self.data), (self,), (1/self.data,))
    def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad

# Initialize the parameters, to store the knowledge of the model
n_layer = 1     # depth of the transformer neural network (number of layers)
n_embd = 16     # width of the network (embedding dimension)
block_size = 16 # maximum context length of the attention window (note: the longest name is 15 characters)
n_head = 4      # number of attention heads
head_dim = n_embd // n_head # derived dimension of each head
matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
state_dict = {'wte': matrix(vocab_size, n_embd), 'wpe': matrix(block_size, n_embd), 'lm_head': matrix(vocab_size, n_embd)}
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)
params = [p for mat in state_dict.values() for row in mat for p in row] # flatten params into a single list[Value]
print(f"num params: {len(params)}")

# Define the model architecture: a function mapping tokens and parameters to logits over what comes next
# Follow GPT-2, blessed among the GPTs, with minor differences: layernorm -> rmsnorm, no biases, GeLU -> ReLU
def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]

def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

def gpt(token_id, pos_id, keys, values):
    tok_emb = state_dict['wte'][token_id] # token embedding
    pos_emb = state_dict['wpe'][pos_id] # position embedding
    x = [t + p for t, p in zip(tok_emb, pos_emb)] # joint token and position embedding
    x = rmsnorm(x) # note: not redundant due to backward pass via the residual connection

    for li in range(n_layer):
        # 1) Multi-head Attention block
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        keys[li].append(k)
        values[li].append(v)
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs+head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]
            attn_weights = softmax(attn_logits)
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]
            x_attn.extend(head_out)
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]
        # 2) MLP block
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]

    logits = linear(x, state_dict['lm_head'])
    return logits

# Let there be Adam, the blessed optimizer and its buffers
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m = [0.0] * len(params) # first moment buffer
v = [0.0] * len(params) # second moment buffer

# Repeat in sequence
num_steps = 1000 # number of training steps
for step in range(num_steps):

    # Take single document, tokenize it, surround it with BOS special token on both sides
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)

    # Forward the token sequence through the model, building up the computation graph all the way to the loss
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    losses = []
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax(logits)
        loss_t = -probs[target_id].log()
        losses.append(loss_t)
    loss = (1 / n) * sum(losses) # final average loss over the document sequence. May yours be low.

    # Backward the loss, calculating the gradients with respect to all model parameters
    loss.backward()

    # Adam optimizer update: update the model parameters based on the corresponding gradients
    lr_t = learning_rate * (1 - step / num_steps) # linear learning rate decay
    for i, p in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
        m_hat = m[i] / (1 - beta1 ** (step + 1))
        v_hat = v[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        p.grad = 0

    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}", end='\r')

# Inference: may the model babble back to us
temperature = 0.5 # in (0, 1], control the "creativity" of generated text, low to high
print("\n--- inference (new, hallucinated names) ---")
for sample_idx in range(20):
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    token_id = BOS
    sample = []
    for pos_id in range(block_size):
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax([l / temperature for l in logits])
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_id == BOS:
            break
        sample.append(uchars[token_id])
    print(f"sample {sample_idx+1:2d}: {''.join(sample)}")
```

---

## 2. Dataset

> _"The fuel of large language models is a stream of text data, optionally separated into a set of documents."_

- 32 000 noms (un par ligne) depuis `names.txt` (Karpathy/makemore)
- Chaque nom = un "document"
- `random.shuffle(docs)` mélange avant entraînement
- Objectif : apprendre les patterns statistiques et générer de nouveaux noms plausibles

Citation clé :

> _"From the perspective of a model like ChatGPT, your conversation with it is just a funny looking 'document'. When you initialize the document with your prompt, the model's response from its perspective is just a statistical document completion."_

---

## 3. Tokenizer

> _"Under the hood, neural networks work with numbers, not characters."_

- Chaque caractère unique (a-z trié) reçoit un ID entier (0–25)
- `BOS = 26` : token spécial "Beginning of Sequence" (début ET fin)
- `vocab_size = 27` (26 lettres + BOS)
- Chaque nom est encadré : `[BOS, e, m, m, a, BOS]`

Citation clé :

> _"Note that the integer values themselves have no meaning at all; each token is just a separate discrete symbol. Instead of 0, 1, 2 they might as well be different emoji."_

Comparaison avec production :

> _"Production tokenizers like tiktoken (used by GPT-4) operate on chunks of characters for efficiency."_

---

## 4. Autograd

> _"Training a neural network requires gradients: for each parameter in the model, we need to know 'if I nudge this number up a little, does the loss go up or down, and by how much?'"_

### Classe Value

Chaque opération :

- Calcule le résultat (forward)
- Enregistre les enfants (`_children`) et les gradients locaux (`_local_grads`)
- `backward()` parcourt le graphe en ordre topologique inverse

### Table des opérations

| Opération | Forward   | Gradients locaux           |
| --------- | --------- | -------------------------- |
| `a + b`   | a + b     | ∂/∂a = 1, ∂/∂b = 1         |
| `a * b`   | a · b     | ∂/∂a = b, ∂/∂b = a         |
| `a ** n`  | a^n       | ∂/∂a = n · a^(n-1)         |
| `log(a)`  | ln(a)     | ∂/∂a = 1/a                 |
| `exp(a)`  | e^a       | ∂/∂a = e^a                 |
| `relu(a)` | max(0, a) | ∂/∂a = 1 si a > 0, 0 sinon |

### Règle de la chaîne

> _"If a car travels twice as fast as a bicycle and the bicycle is four times as fast as a walking man, then the car travels 2 × 4 = 8 times as fast as the man."_

Formule : `∂L/∂c += (∂v/∂c) · (∂L/∂v)`

Le `+=` (accumulation) est essentiel : si une valeur contribue à L par plusieurs chemins, les gradients de chaque chemin sont sommés.

### Exemple concret

```python
a = Value(2.0)
b = Value(3.0)
c = a * b       # c = 6.0
L = c + a       # L = 8.0
L.backward()
print(a.grad)   # 4.0 (dL/da = b + 1 = 3 + 1, via les deux chemins)
print(b.grad)   # 2.0 (dL/db = a = 2)
```

Citation clé :

> _"This is the same algorithm that PyTorch's `loss.backward()` runs, just on scalars instead of tensors — algorithmically identical, significantly smaller and simpler, but of course a lot less efficient."_

---

## 5. Parameters

> _"The parameters are the knowledge of the model."_

### Hyperparamètres

| Paramètre    | Valeur | Description                           |
| ------------ | ------ | ------------------------------------- |
| `n_embd`     | 16     | Dimension des plongements             |
| `n_head`     | 4      | Nombre de têtes d'attention           |
| `n_layer`    | 1      | Nombre de couches transformer         |
| `block_size` | 16     | Longueur max de séquence              |
| `head_dim`   | 4      | Dimension par tête (n_embd // n_head) |

### Matrices du state_dict

| Clé              | Dimensions | Paramètres |
| ---------------- | ---------- | ---------- |
| `wte`            | 27 × 16    | 432        |
| `wpe`            | 16 × 16    | 256        |
| `lm_head`        | 27 × 16    | 432        |
| `layer0.attn_wq` | 16 × 16    | 256        |
| `layer0.attn_wk` | 16 × 16    | 256        |
| `layer0.attn_wv` | 16 × 16    | 256        |
| `layer0.attn_wo` | 16 × 16    | 256        |
| `layer0.mlp_fc1` | 64 × 16    | 1 024      |
| `layer0.mlp_fc2` | 16 × 64    | 1 024      |
| **Total**        |            | **4 192**  |

Initialisation : gaussienne, σ = 0.08

> _"GPT-2 had 1.6 billion, and modern LLMs have hundreds of billions."_

---

## 6. Architecture

> _"The model architecture is a stateless function: it takes a token, a position, the parameters, and the cached keys/values from previous positions, and returns logits."_
>
> _"We follow GPT-2 with minor simplifications: RMSNorm instead of LayerNorm, no biases, and ReLU instead of GeLU."_

### Fonctions utilitaires

**`linear(x, w)`** — Multiplication matrice-vecteur

> _"This is the fundamental building block of neural networks: a learned linear transformation."_

**`softmax(logits)`** — Scores → probabilités [0,1] sommant à 1

> _"We subtract the max first for numerical stability (it doesn't change the result mathematically, but prevents overflow in exp)."_

**`rmsnorm(x)`** — Normalisation RMS (Root Mean Square)

> _"This keeps activations from growing or shrinking as they flow through the network, which stabilizes training. It's a simpler variant of the LayerNorm used in the original GPT-2."_

### Fonction `gpt(token_id, pos_id, keys, values)`

**Pipeline complet (un token à la fois) :**

```
token_id, pos_id
        │
        ▼
┌─── Embeddings ───┐
│ tok_emb = wte[token_id]
│ pos_emb = wpe[pos_id]
│ x = tok_emb + pos_emb
│ x = rmsnorm(x)
└──────────────────┘
        │
        ▼
┌── Attention block ──┐
│ x_residual = x      │
│ x = rmsnorm(x)      │
│ Q = linear(x, wq)   │
│ K = linear(x, wk)   │
│ V = linear(x, wv)   │
│ keys[li].append(K)   │
│ values[li].append(V) │
│                      │
│ Pour chaque tête h:  │
│   q_h, k_h, v_h     │
│   attn = softmax(    │
│     QK^T / √d_head)  │
│   out_h = attn · V   │
│                      │
│ x = linear(concat,wo)│
│ x = x + x_residual   │ ← connexion résiduelle
└──────────────────────┘
        │
        ▼
┌──── MLP block ────┐
│ x_residual = x    │
│ x = rmsnorm(x)    │
│ x = linear(x,fc1) │  16 → 64
│ x = relu(x)       │
│ x = linear(x,fc2) │  64 → 16
│ x = x + x_residual│ ← connexion résiduelle
└────────────────────┘
        │
        ▼
┌──── Output ────┐
│ logits =       │
│  linear(x,     │
│   lm_head)     │  16 → 27
└────────────────┘
```

### Explications clés de Karpathy

**Embeddings :**

> _"The token id and position id each look up a row from their respective embedding tables (wte and wpe). These two vectors are added together, giving the model a representation that encodes both **what** the token is and **where** it is in the sequence."_

**Attention (citations intégrales — concepts les plus importants du guide) :**

> _"The current token is projected into three vectors: a query (Q), a key (K), and a value (V). Intuitively, the query says 'what am I looking for?', the key says 'what do I contain?', and the value says 'what do I offer if selected?'."_
>
> _"For example, in the name 'emma', when the model is at the second 'm' and trying to predict what comes next, it might learn a query like 'what vowels appeared recently?' The earlier 'e' would have a key that matches this query well, so it gets a high attention weight, and its value (information about being a vowel) flows into the current position."_
>
> _"The key and value are appended to the KV cache so previous positions are available."_
>
> _"**It's worth emphasizing that the Attention block is the exact and only place where a token at position t gets to 'look' at tokens in the past 0..t-1. Attention is a token communication mechanism.**"_

**MLP :**

> _"MLP is short for 'multilayer perceptron', it is a two-layer feed-forward network: project up to 4x the embedding dimension, apply ReLU, project back down. This is where the model does most of its 'thinking' per position. Unlike attention, this computation is fully local to time t."_
>
> _"**The Transformer intersperses communication (Attention) with computation (MLP).**"_

**Connexions résiduelles :**

> _"Both the attention and MLP blocks add their output back to their input (`x = [a + b for ...]`). This lets gradients flow directly through the network and makes deeper models trainable."_

**KV Cache pendant l'entraînement :**

> _"You might notice that we're using a KV cache during training, which is unusual. [...] Since microgpt processes one token at a time (no batch dimension, no parallel time steps), we build the KV cache explicitly. And unlike the typical inference setting where the KV cache holds detached tensors, here the cached keys and values are live Value nodes in the computation graph, so we actually backpropagate through them."_

---

## 7. Training loop

> _"The training loop repeatedly: (1) picks a document, (2) runs the model forward over its tokens, (3) computes a loss, (4) backpropagates to get gradients, and (5) updates the parameters."_

### Tokenisation du document

Chaque étape prend un nom et l'encadre de BOS : `"emma"` → `[BOS, e, m, m, a, BOS]`

### Forward pass et loss

> _"At each position, the model outputs 27 logits, which we convert to probabilities via softmax. The loss at each position is the negative log probability of the correct next token: `-log p(target)`. This is called the cross-entropy loss."_
>
> _"Intuitively, the loss measures the degree of misprediction: how surprised the model is by what actually comes next. If the model assigns probability 1.0 to the correct token, it is not surprised at all and the loss is 0. If it assigns probability close to 0, the model is very surprised and the loss goes to +∞."_

### Backward pass

> _"One call to `loss.backward()` runs backpropagation through the entire computation graph, from the loss all the way back through softmax, the model, and into every parameter. After this, each parameter's `.grad` tells us how to change it to reduce the loss."_

### Adam optimizer

| Paramètre       | Valeur             |
| --------------- | ------------------ |
| `learning_rate` | 0.01               |
| `beta1`         | 0.85               |
| `beta2`         | 0.99               |
| `eps_adam`      | 1e-8               |
| LR decay        | Linéaire jusqu'à 0 |
| `num_steps`     | 1 000              |

> _"Adam is smarter [than SGD]. It maintains two running averages per parameter: m tracks the mean of recent gradients (momentum, like a rolling ball), and v tracks the mean of recent squared gradients (adapting the learning rate per parameter)."_

### Résultats

- Loss initiale : ~3.3 (devinette aléatoire, `-log(1/27) ≈ 3.3`)
- Loss finale : ~2.37 après 1 000 étapes

---

## 8. Inference

> _"Once training is done, we can sample new names from the model. The parameters are frozen and we just run the forward pass in a loop, feeding each generated token back as the next input."_

### Température

> _"A temperature of 1.0 samples directly from the model's learned distribution. Lower temperatures (like 0.5 here) sharpen the distribution, making the model more conservative and likely to pick its top choices. A temperature approaching 0 would always pick the single most likely token (greedy decoding). Higher temperatures flatten the distribution and produce more diverse but potentially less coherent output."_

### Échantillonnage

Logits ÷ temperature → softmax → `random.choices` (weighted sampling) → si BOS → stop

### Exemples de sortie (température 0.5)

```
sample  1: kamon      sample 11: konna
sample  2: ann        sample 12: keylen
sample  3: karai      sample 13: liole
sample  4: jaire      sample 14: alerin
sample  5: vialan     sample 15: earan
sample  6: karia      sample 16: lenne
sample  7: yeran      sample 17: kana
sample  8: anna       sample 18: lara
sample  9: areli      sample 19: alela
sample 10: kaina      sample 20: anton
```

---

## 9. Progression

Couches successives du code, de simple à complet :

| Fichier     | Ce qu'il ajoute                                                      |
| ----------- | -------------------------------------------------------------------- |
| `train0.py` | Table bigrammes — pas de réseau, pas de gradients                    |
| `train1.py` | MLP + gradients manuels (numérique & analytique) + SGD               |
| `train2.py` | Autograd (classe Value) — remplace les gradients manuels             |
| `train3.py` | Plongements de position + attention à une tête + rmsnorm + résidus   |
| `train4.py` | Attention multi-têtes + boucle de couche — architecture GPT complète |
| `train5.py` | Optimiseur Adam — c'est `train.py` (= microgpt.py)                   |

Source : <https://gist.github.com/karpathy/561ac2de12a47cc06a23691e1be9543a>

---

## 10. Real stuff

Différences entre microgpt et les LLM de production (ChatGPT, etc.) :

| Aspect            | microgpt                   | Production                                          |
| ----------------- | -------------------------- | --------------------------------------------------- |
| **Data**          | 32K noms courts            | Trillions de tokens (web, livres, code)             |
| **Tokenizer**     | 1 char = 1 token (27)      | BPE subword (~100K tokens)                          |
| **Autograd**      | Scalaires Python (`Value`) | Tenseurs GPU (PyTorch, FlashAttention)              |
| **Architecture**  | 4 192 params, 1 couche     | Milliards de params, 100+ couches, RoPE, GQA, MoE   |
| **Training**      | 1 doc/step, 1 000 steps    | Millions de tokens/step, mois de calcul GPU         |
| **Optimizer**     | Adam + LR decay linéaire   | Mixed precision, scaling laws, hyperparams tunés    |
| **Post-training** | Aucun                      | SFT + RLHF                                          |
| **Inference**     | Python séquentiel          | vLLM, speculative decoding, quantization, multi-GPU |

Citation clé :

> _"All of these are important engineering and research contributions but if you understand microgpt, you understand the algorithmic essence."_

Le Transformer en production :

> _"The core structure of Attention (communication) and MLP (computation) interspersed on a residual stream is well-preserved."_

---

## 11. FAQ

**"Does the model 'understand' anything?"**

> _"Mechanically: no magic is happening. The model is a big math function that maps input tokens to a probability distribution over the next token."_

**"Why does it work?"**

> _"The model doesn't learn explicit rules, it learns a probability distribution that happens to reflect them."_

**"How is this related to ChatGPT?"**

> _"ChatGPT is this same core loop (predict next token, sample, repeat) scaled up enormously, with post-training to make it conversational."_

**"What's the deal with 'hallucinations'?"**

> _"The model generates tokens by sampling from a probability distribution. It has no concept of truth, it only knows what sequences are statistically plausible."_
>
> _"microgpt 'hallucinating' a name like 'karia' is the same phenomenon as ChatGPT confidently stating a false fact."_

**"Can I make it generate better names?"**

> _"Train longer (increase num_steps), make the model bigger (n_embd, n_layer, n_head), or use a larger dataset. These are the same knobs that matter at scale."_

---

## 12. Correspondance guide ↔ app

### Mapping sections du guide → pages de l'app

| Section guide                | Page app               | Fichier source           | Fidélité                                              |
| ---------------------------- | ---------------------- | ------------------------ | ----------------------------------------------------- |
| 2. Dataset                   | Sidebar (sélecteur)    | `App.tsx`, `datasets/`   | Adapté (6 datasets FR au lieu de names.txt)           |
| 3. Tokenizer                 | 1. Tokenisation        | `TokenizerPage.tsx`      | Fidèle                                                |
| 4. Autograd                  | (pas de page)          | `engine/autograd.ts`     | Moteur fidèle, pas visualisé (choix public 10-14 ans) |
| 5. Parameters                | (visible dans page 2)  | `engine/model.ts:59-96`  | Fidèle (badge "aléatoire")                            |
| 6. Architecture — embeddings | 2. Plongements         | `EmbeddingsPage.tsx`     | Fidèle                                                |
| 6. Architecture — attention  | 3. Propagation         | `ForwardPassPage.tsx`    | **DÉFAILLANT** : token unique → `[1.0]`               |
| 6. Architecture — MLP        | 3. Propagation         | `MLPActivationPanel.tsx` | Fidèle                                                |
| 6. Architecture — résidus    | 3. Propagation         | `VectorsPanel.tsx`       | Partiel (montré, pas expliqué)                        |
| 7. Training loop             | 4. Entraînement        | `TrainingPage.tsx`       | Fidèle (loss live + détail)                           |
| 8. Inference                 | 5. Inférence           | `InferencePage.tsx`      | Fidèle (trace + température)                          |
| 9. Progression               | Playgrounds standalone | `playground*.html`       | Adapté                                                |
| 10. Real stuff               | TokenizerPage (BPE)    | (comparaison inline)     | Partiel                                               |

### Mapping model.ts → microgpt.py (port exact)

| microgpt.py                                     | model.ts                                                 | Identique ?             |
| ----------------------------------------------- | -------------------------------------------------------- | ----------------------- |
| `class Value`                                   | `class Value` (autograd.ts)                              | Oui                     |
| `uchars, BOS, vocab_size`                       | `uchars, BOS, vocabSize`                                 | Oui                     |
| `n_embd=16, n_head=4, n_layer=1, block_size=16` | `N_EMBD=16, N_HEAD=4, N_LAYER=1, BLOCK_SIZE=16`          | Oui                     |
| `matrix(nout, nin, std=0.08)`                   | `makeMatrix(nout, nin, rng, std=0.08)`                   | Oui (+ PRNG injectable) |
| `state_dict` (9 matrices)                       | `stateDict` (9 matrices)                                 | Oui                     |
| `linear(x, w)`                                  | `linear(x, w)`                                           | Oui                     |
| `softmax(logits)`                               | `softmax(logits)`                                        | Oui                     |
| `rmsnorm(x)`                                    | `rmsnorm(x)`                                             | Oui                     |
| `gpt(token_id, pos_id, keys, values)`           | `gptForward(tokenId, posId, keys, values, state, trace)` | Oui (+ trace optionnel) |
| Training: Adam β1=0.85 β2=0.99                  | `trainStep`: mêmes constantes                            | Oui                     |
| Inference: temperature + sampling               | `generateName`: même logique                             | Oui                     |

### Citation clé pour le gap attention

Le guide dit explicitement :

> **"It's worth emphasizing that the Attention block is the exact and only place where a token at position t gets to 'look' at tokens in the past 0..t-1. Attention is a token communication mechanism."**

L'app actuelle montre un token isolé (t=0, KV cache vide) → aucune communication possible → l'attention affiche `[1.0]` → contresens pédagogique.

---

## 13. Déviations intentionnelles — microgpt-lab vs Karpathy

Ce projet est un playground d'expérimentation, pas un port fidèle. Certains choix s'écartent volontairement du code original pour des raisons UX. Chaque déviation est documentée ici.

### 13.1 LR schedule : constant au lieu de linear decay

| | Karpathy (`microgpt.py`) | microgpt-lab |
|---|---|---|
| **Formule** | `lr_t = lr * (1 - step / num_steps)` | `lr_t = lr` (constant) |
| **Code** | ligne 215 de microgpt.py | `adam_step()` dans `model.rs` et `tensor_model.rs` |
| **Comportement** | LR décroît linéairement de `lr` à 0 sur `num_steps` | LR reste fixe, contrôlé par l'utilisateur via slider |

**Pourquoi Karpathy utilise le decay :**
- Training one-shot de 1000 steps sur un dataset fixe (32K noms)
- Le decay permet au modèle de "se poser" au minimum en fin de training
- `num_steps` est connu à l'avance et fixe

**Pourquoi microgpt-lab dévie :**
- Le playground permet de lancer des batches successifs (200, 500, 2000 steps)
- Avec le decay, `lr_t = 0` après `num_steps` (hardcodé à 1000) → les poids sont gelés, tout batch ultérieur est inutile
- L'UX est "tweak & retry" : l'utilisateur expérimente librement, pas un training one-shot
- Le slider LR donne le contrôle explicite — pas de comportement caché

**Impact ML :** Négligeable sur des modèles de cette taille (4192 params). La convergence fine que le decay apporte n'est pas observable sur un toy model qui overfit de toute façon.

**Date :** 2026-03-14 — Task 61
