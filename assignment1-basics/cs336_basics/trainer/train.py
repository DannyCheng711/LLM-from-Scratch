import argparse
import time
from pathlib import Path

import numpy as np
import torch
from einx import check
from torch.utils.checkpoint import checkpoint

from cs336_basics.model_architecture.transformer_lm import TransformerLM
from cs336_basics.trainer.dataloader import get_batch
from cs336_basics.trainer.cross_entropy import cross_entropy
from cs336_basics.trainer.adamw import AdamW
from cs336_basics.trainer.lr_scheduler import get_lr_cosine_schedule
from cs336_basics.trainer.gradient_clipping import get_gradient_clip
from cs336_basics.trainer.checkpoint import (
    save_checkpoint, load_checkpoint,
)

@torch.no_grad()
def evaluate(
    model, dataset, batch_size, context_length, device, num_batches
):
    """
    Estimate validation loss over several randomly sampled batches.
    """

    model.eval()
    total_loss = 0.0

    for _ in range(num_batches):
        inputs, targets = get_batch(
            x = dataset,
            batch_size= batch_size,
            context_length= context_length,
            device = device,
        )

        logits = model(inputs)
        loss = cross_entropy(logits, targets) # targets would become target logits in this func

        total_loss += loss.item()

    model.train() # switch to training mode

    return total_loss / num_batches

def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = args.device

    # =====================================
    # 1. Load Datasets with Memory Mapping
    # =====================================
    train_data = np.load(
        args.train_data,
        mmap_mode= "r", # lazy loader
    )

    val_data = np.load(
        args.val_data,
        mmap_mode= "r"
    )

    print(f"Train tokens: {len(train_data):,}")
    print(f"Validation tokens: {len(val_data):,}")
    print(f"Device: {device}")

    # =====================================
    # 2. Build model
    # =====================================

    model = TransformerLM(
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        vocab_size= args.vocab_size,
        max_seq_len=args.context_length,
        theta=args.rope_theta,
        context_length= args.context_length,
        num_layers = args.num_layers,
        device = device,
        dtype = torch.float32,
    )

    model.train()

    # =====================================
    # 3. Build optimizer
    # =====================================
    optimizer = AdamW(
        params=model.parameters(),
        lr = args.max_lr,
        betas = (args.beta1, args.beta2),
        eps = args.eps,
        weight_decay= args.weight_decay
    )

    # =====================================
    # 4. Resume from checkpoint
    # =====================================
    start_iteration = 0

    if args.resume_from is not None:
        start_iteration = load_checkpoint(
            src=args.resume_from,
            model=model,
            optimizer=optimizer,
        )

        print(
            f"Resumed from {args.resume_from} at iteration {start_iteration}"
        )



    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    # =====================================
    # 5. Training loop
    # =====================================

    for iteration in range(start_iteration, args.max_iterations):
        # Compute this iteration's learning rate.
        lr = get_lr_cosine_schedule(
            t=iteration,
            alpha_max=args.max_lr,
            alpha_min=args.min_lr,
            T_w=args.warmup_iterations,
            T_c=args.cosine_iterations,
        )

        # Update the optimizer's current LR.
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

            # Sample one training batch.

            inputs, targets = get_batch(
                x=train_data,
                batch_size=args.batch_size,
                context_length=args.context_length,
                device=device,
            )

            # Clear gradients from the previous iteration.
            optimizer.zero_grad()

            # Forward pass.
            logits = model(inputs)

            # logits:  (batch, seq, vocab_size)
            # targets: (batch, seq)
            loss = cross_entropy(logits, targets)

            # Backward pass.
            loss.backward()

            # Clip the combined model gradient norm.
            get_gradient_clip(
                parameters=model.parameters(),
                max_l2_norm=args.max_grad_norm,
            )

            # Update model params
            optimizer.step()

            completed_iteration = iteration + 1

            # =================================
            # Logging
            # =================================
            if completed_iteration % args.log_interval == 0:
                elapsed = time.time() - start_time
                print(
                    f"Iteration {completed_iteration:>7} | "
                    f"Train Loss {loss.item():.4f} | "
                    f"LR {lr:.6e} | "
                    f"Elapsed {elapsed:.1f}s"
                )

            # =================================
            # Validation
            # =================================

            if completed_iteration % args.eval_interval == 0:
                val_loss = evaluate(
                    model=model,
                    dataset=val_data,
                    batch_size=args.batch_size,
                    context_length=args.context_length,
                    device=device,
                    num_batches=args.eval_batches,
                )

                print(
                    f"Iteration {completed_iteration:>7} | "
                    f"Validation Loss {val_loss:.4f}"
                )

            # ============================
            # Checkpoint
            # ============================

            if completed_iteration % args.checkpoint_interval == 0:
                checkpoint_path = (
                    checkpoint_dir
                        / f"checkpoint_{completed_iteration}.pt"
                )

                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    iteration=completed_iteration,
                    out=checkpoint_path,
                )

                print(f"Saved checkpoint: {checkpoint_path}")



    # =====================================
    # 6. Save final checkpoint
    # =====================================
    final_checkpoint = checkpoint_dir / "checkpoint_final.pt"
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        iteration=args.max_iterations,
        out=final_checkpoint
    )

    print(f"Training complete. Final checkpoint: {final_checkpoint}")


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="Train a Transformer language model."
    )

    # Data
    parser.add_argument("--train-data", type=str, required=True)
    parser.add_argument("--val-data", type=str, required=True)

    # Model
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--rope-theta", type=float, default=10000.0)

    # Optimization
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-iterations", type=int, default=10000)
    parser.add_argument("--max-lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-iterations", type=int, default=500)
    parser.add_argument("--cosine-iterations", type=int, default=10000)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    # Logging and checkpointing
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--resume-from", type=str, default=None)

    # Runtime
    parser.add_argument(
        "--device",
        type=str,
        default=(
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )

    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()

if __name__ == "__main__":
    train(parse_args())

