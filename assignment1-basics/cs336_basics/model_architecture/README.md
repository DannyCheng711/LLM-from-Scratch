### Transformer LM Parameter Count

Consider GPT-2 XL with the following configuration:

- Vocabulary size: 50,257
- Context length: 1,024
- Number of layers: 48
- Model dimension (`d_model`): 1,600
- Number of attention heads: 25
- Feed-forward dimension (`d_ff`): 6,400

How many trainable parameters does the model have? 

Assuming each parameter is stored as a single-precision floating-point value (FP32), how much memory is required to load the model?

---

### A. Memory Usage 

1. Token Embedding: Embedding matrix = vocab size * d_model = 50,257 * 1,600 = 80,411,200
2. One Transformer Block: 40,963,200
   1. Q, K, V, output: 4 * d_model * d_model = 10,240,000
   2. RMSNorm: ln1 , ln2 : 2 * d_model = 3,200
   3. Feed Forward: W1 (d_model -> d_ff), W2 (d_ff->d_model), W3 (d_model -> d_ff): 3 * 1,600 * 6,400 = 30,720,000
3. 48 layers: 48 * 40,963,200 = 1,966,233,600
4. Final RMSNorm: d_model = 1,600
5. LM Head: d_modex * V = 1,600 * 50,257 = 80,411,200

Sum = 80,411,200 + 1,966,233,600 + 1,600 + 80,411,200 = $\boxed{2.13\times10^9}$
Memory (float32) = $\boxed{2.13\times10^9} * 4 = 8.51GB$ 

---

### B. FLOPs for GPT-2 XL Forward Pass

Configuration:

- Sequence length \(n=1024\)
- Layers \(L=48\)
- Model dimension \(d=1600\)
- Heads \(h=25\)
- Head dimension \(d_h=64\)
- FFN dimension \(d_{ff}=6400\)
- Vocabulary size \(V=50257\)

### Matrix multiplies per Transformer layer

1. Q/K/V projections: $3 \times 2n * d^2 = 15.73\text{B FLOPs}$
2. Attention scores $QK^\top$: $2n^2d =  3.36\text{B FLOPs}$
3. Attention-value product $AV$: $ 2n^2d =  3.36\text{B FLOPs}$
4. Attention output projection: $2n * d^2 = 5.24\text{B FLOPs}$
5. FFN / SwiGLU projections: $3 \times 2n * d * d_{ff} = 62.91\text{B FLOPs}$

- Sum = $90.60\text{B FLOPs}$
- 48 Layers =  $48 \times 90.60\text{B}$ = $4.35\text{T FLOPs}$
- Final LM head = $2n * d * V = 164.68\text{B FLOPs}$

- Total = $4.35\text{T FLOPs} + 1.65 \text{T FLOPs} = {4.51\text{T FLOPs}}$ 

--- 

### C. Components require the most FLOPs

- The feed-forward network (FFN) requires the most FLOPs, 
because each layer performs three large matrix multiplications involving $d_{model}$ and $d_{ff}$. 

---

### D. Different Size GPT-2

Assume context length n=1024, vocabulary size V=50257, and $d_{ff}=4d_{model}$.

| Model | Total FLOPs | QKV Proj | QKᵀ | AV | Output Proj | FFN / SwiGLU | LM Head |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GPT-2 Small | **0.350T** | 12.4% | 5.5% | 5.5% | 4.1% | **49.8%** | 22.6% |
| GPT-2 Medium | **1.033T** | 15.0% | 5.0% | 5.0% | 5.0% | **59.9%** | 10.2% |
| GPT-2 Large | **2.258T** | 16.1% | 4.3% | 4.3% | 5.4% | **64.2%** | 5.8% |
| GPT-2 XL | **4.513T** | 16.7% | 3.6% | 3.6% | 5.6% | **66.9%** | 3.6% |

- As model size increases, the FFN takes up a larger proportion of total FLOPs because it scales strongly with $d_{model}$, $d_{ff}$. 
- The LM head becomes proportionally smaller, while attention score computation becomes less dominant at fixed context length.

--- 

### E. GPT-2 XL with Context Length 16,384

Increasing GPT-2 XL context length from 1,024 to 16,384 increases the total forward-pass FLOPs from about **4.51T** to about **149.52T**.