### A. Transformer LM Parameter Count

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

### Solution 

1. Token Embedding: Embedding matrix = vocab size * d_model = 50,257 * 1,600 = 80,411,200
2. One Transformer Block: 40,963,200
   1. Q, K, V, output: 4 * d_model * d_model = 10,240,000
   2. RMSNorm: ln1 , ln2 : 2 * d_model = 3,200
   3. Feed Forward: W1 (d_model -> d_ff), W2 (d_ff->d_model), W3 (d_model -> d_ff): 3 * 1,600 * 6,400 = 30,720,000
3. 48 layers: 48 * 40,963,200 = 1,966,233,600
4. Final RMSNorm: d_model = 1,600
5. LM Head: d_modex * V = 1,600 * 50,257 = 80,411,200

Sum = 80,411,200 + 1,966,233,600 + 1,600 + 80,411,200 = $\boxed{2.13\times10^9}$
