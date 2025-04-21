import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

from Transformer import logger
from Transformer.entity.config_entity import ModelTrainingConfig


class TranslationDataset(Dataset):
    """Dataset for machine translation"""

    def __init__(self, data_path, split_type="train"):
        """
        Initialize the translation dataset.

        Args:
            data_path: Directory containing preprocessed data
            split_type: Dataset split (train, valid)
        """
        self.data_path = data_path
        self.split_type = split_type

        # Load preprocessed data
        self.df = pd.read_csv(os.path.join(data_path, f"{split_type}.csv"))

        # Convert string representation of lists to actual lists
        self.df["de"] = self.df["de"].apply(eval)
        self.df["en"] = self.df["en"].apply(eval)

        logger.info(f"Loaded {split_type} dataset: {len(self.df)} samples")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """Get source and target sequences by index"""
        item = self.df.iloc[idx]
        source = torch.tensor(item["de"], dtype=torch.long)
        target = torch.tensor(item["en"], dtype=torch.long)
        return {"source": source, "target": target}


class TokenBucketSampler(Sampler):
    """
    Sampler that groups sequences of similar lengths together to minimize padding.
    This implementation batches by approximate number of tokens rather than sequences.
    """

    def __init__(
        self, dataset, src_tokens_per_batch, tgt_tokens_per_batch, shuffle=True
    ):
        """
        Initialize the token bucket sampler.

        Args:
            dataset: The translation dataset
            src_tokens_per_batch: Max source tokens per batch
            tgt_tokens_per_batch: Max target tokens per batch
            shuffle: Whether to shuffle batches
        """
        self.dataset = dataset
        self.src_tokens_per_batch = src_tokens_per_batch
        self.tgt_tokens_per_batch = tgt_tokens_per_batch
        self.shuffle = shuffle

        # Get sequence lengths
        self.src_lens = np.array(
            [len(self.dataset.df.iloc[i]["de"]) for i in range(len(dataset))]
        )
        self.tgt_lens = np.array(
            [len(self.dataset.df.iloc[i]["en"]) for i in range(len(dataset))]
        )

        # Sort indices by sequence length
        self.sorted_indices = np.argsort(self.src_lens)

        # Create batches
        self.batches = self._create_batches()

    def _create_batches(self):
        """Group sequences into batches based on length"""
        batches = []
        indices = self.sorted_indices.copy()

        # Create batches by iterating through sorted indices
        batch_indices = []
        src_batch_tokens = 0
        tgt_batch_tokens = 0
        max_src_len = 0
        max_tgt_len = 0

        for idx in indices:
            src_len = self.src_lens[idx]
            tgt_len = self.tgt_lens[idx]

            # Check if adding this sequence would exceed token limits
            if (
                max(max_src_len, src_len) * (len(batch_indices) + 1)
                > self.src_tokens_per_batch
                or max(max_tgt_len, tgt_len) * (len(batch_indices) + 1)
                > self.tgt_tokens_per_batch
            ):
                # Current batch is full, add to batches
                if batch_indices:
                    batches.append(batch_indices)

                # Start a new batch
                batch_indices = [idx]
                max_src_len = src_len
                max_tgt_len = tgt_len
            else:
                # Add to current batch
                batch_indices.append(idx)
                max_src_len = max(max_src_len, src_len)
                max_tgt_len = max(max_tgt_len, tgt_len)

        # Add the last batch if it exists
        if batch_indices:
            batches.append(batch_indices)

        return batches

    def __iter__(self):
        """Iterate through batches"""
        if self.shuffle:
            np.random.shuffle(self.batches)

        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)


def create_padding_mask(seq, pad_idx=0):
    """
    Creates a boolean mask to hide padding tokens.

    Args:
        seq: Input sequence tensor [batch_size, seq_len]
        pad_idx: Padding token index

    Returns:
        mask: Boolean mask with True for non-padding positions [batch_size, 1, 1, seq_len]
    """
    # seq: [batch_size, seq_len]
    mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
    return mask


def create_look_ahead_mask(size):
    """
    Creates a look-ahead mask for decoder self-attention.

    Args:
        size: Sequence length

    Returns:
        mask: Lower triangular mask to prevent attending to future positions
    """
    # Create a lower triangular mask (including diagonal)
    mask = torch.tril(torch.ones(size, size))
    return mask


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing loss as described in the paper.
    Mixes the one-hot distribution with a uniform distribution.
    """

    def __init__(self, smoothing=0.1, ignore_index=0, reduction="mean"):
        """
        Initialize label smoothing loss.

        Args:
            smoothing: Smoothing factor
            ignore_index: Index to ignore (usually padding)
            reduction: Reduction method ('mean', 'sum', or 'none')
        """
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, output, target):
        """
        Calculate label smoothing loss.

        Args:
            output: Model output logits [batch_size, seq_len, vocab_size]
            target: Target indices [batch_size, seq_len]

        Returns:
            loss: Smoothed cross-entropy loss
        """
        log_prob = F.log_softmax(output, dim=-1)

        # Create a mask for non-padding positions
        non_pad_mask = (target != self.ignore_index).float()

        # Get the number of non-padding tokens
        n_tokens = non_pad_mask.sum()

        # Gather the log probabilities of the target tokens
        target_log_prob = log_prob.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(
            -1
        )

        # Apply label smoothing
        # Instead of using all probability mass on the correct token,
        # we distribute (1-smoothing) to the correct token and smoothing/vocab_size to all tokens
        smoothed_target_log_prob = (
            1.0 - self.smoothing
        ) * target_log_prob + self.smoothing * log_prob.mean(dim=-1)

        # Apply the mask to ignore padding tokens
        loss = -smoothed_target_log_prob * non_pad_mask

        # Apply reduction
        if self.reduction == "none":
            return loss
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'mean'
            return loss.sum() / n_tokens


class NoamLR:
    """
    Learning rate scheduler with warmup as described in the paper:
    lrate = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
    """

    def __init__(self, optimizer, d_model, warmup_steps, factor=1.0):
        """
        Initialize the learning rate scheduler.

        Args:
            optimizer: Optimizer instance
            d_model: Model dimension
            warmup_steps: Number of warmup steps
            factor: Multiplicative factor
        """
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self._step = 0
        self._rate = 0

    def step(self):
        """Update learning rate and take an optimization step"""
        self._step += 1
        rate = self.get_rate()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = rate
        self._rate = rate

    def get_rate(self):
        """Calculate the learning rate based on current step"""
        step = max(1, self._step)
        # Implementation of the formula from the paper
        return (
            self.factor
            * (self.d_model**-0.5)
            * min(step**-0.5, step * (self.warmup_steps**-1.5))
        )


class ModelTrainer:
    def __init__(self, config: ModelTrainingConfig):
        """
        Initialize the model trainer.

        Args:
            config: Model training configuration
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Load tokenizer
        tokenizer_path = os.path.join(str(config.data_path.parent), "bpe_tokenizer")
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=os.path.join(tokenizer_path, "tokenizer.json"),
            pad_token="<pad>",
            unk_token="<unk>",
            bos_token="<s>",
            eos_token="</s>",
            cls_token="<s>",
            sep_token="</s>",
            mask_token="<mask>",
        )

        # Special token IDs
        self.pad_id = self.tokenizer.pad_token_id
        self.bos_id = self.tokenizer.bos_token_id
        self.eos_id = self.tokenizer.eos_token_id

        # Load model
        self.load_model()

        # Load datasets
        self.load_datasets()

        # Initialize WandB if specified
        if self.config.wandb_project_name:
            wandb.init(
                project=self.config.wandb_project_name,
                name=self.config.wandb_run_name,
                config={
                    "warmup_steps": self.config.warmup_steps,
                    "src_tokens_per_batch": self.config.src_tokens_per_batch,
                    "tgt_tokens_per_batch": self.config.tgt_tokens_per_batch,
                    "total_steps": self.config.total_steps,
                    "lr_factor": self.config.lr_factor,
                    "clip_grad": self.config.clip_grad,
                    "adam_beta1": self.config.adam_beta1,
                    "adam_beta2": self.config.adam_beta2,
                    "adam_epsilon": self.config.adam_epsilon,
                    "label_smoothing": self.config.label_smoothing,
                    "last_n_checkpoints_to_avg": self.config.last_n_checkpoints_to_avg,
                    "checkpoint_interval_minutes": self.config.checkpoint_interval_minutes,
                    "beam_size": self.config.beam_size,
                    "length_penalty": self.config.length_penalty,
                    "max_length": self.config.max_length,
                    "num_layers": self.config.num_layers,
                    "vocab_size": self.config.vocab_size,
                    "d_model": self.config.d_model,
                    "dropout": self.config.dropout,
                    "num_heads": self.config.num_heads,
                    "dff": self.config.dff,
                },
            )
            wandb.watch(self.model)

        # Initialize optimizer and scheduler
        self.initialize_training()

    def load_model(self):
        """Load the Transformer model"""
        from Transformer.components.build_model import Transformer

        self.model = Transformer(
            num_layers=self.config.num_layers,
            d_model=self.config.d_model,
            num_heads=self.config.num_heads,
            d_ff=self.config.dff,
            vocab_size=self.tokenizer.vocab_size,
            dropout=self.config.dropout,
            max_len=self.config.max_length,
        )
        self.model.load_state_dict(
            torch.load(self.config.model_path, weights_only=False)
        )
        self.model = self.model.to(self.device)

    def load_datasets(self):
        """Load and prepare datasets"""
        # Create datasets
        self.train_dataset = TranslationDataset(self.config.data_path, "train")
        self.valid_dataset = TranslationDataset(self.config.data_path, "valid")

        # Create samplers
        self.train_sampler = TokenBucketSampler(
            self.train_dataset,
            self.config.src_tokens_per_batch,
            self.config.tgt_tokens_per_batch,
            shuffle=True,
        )

        self.valid_sampler = TokenBucketSampler(
            self.valid_dataset,
            self.config.src_tokens_per_batch,
            self.config.tgt_tokens_per_batch,
            shuffle=False,
        )

        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_sampler=self.train_sampler,
            collate_fn=self.collate_fn,
        )

        self.valid_loader = DataLoader(
            self.valid_dataset,
            batch_sampler=self.valid_sampler,
            collate_fn=self.collate_fn,
        )

    def collate_fn(self, batch):
        """
        Custom collate function to handle variable length sequences.

        Args:
            batch: List of samples

        Returns:
            Padded batch dictionary
        """
        # Extract source and target sequences
        source_seqs = [item["source"] for item in batch]
        target_seqs = [item["target"] for item in batch]

        # Get max lengths
        src_max_len = max(len(seq) for seq in source_seqs)
        tgt_max_len = max(len(seq) for seq in target_seqs)

        # Pad sequences
        padded_sources = []
        padded_targets = []

        for src, tgt in zip(source_seqs, target_seqs):
            # Pad source
            padded_src = F.pad(src, (0, src_max_len - len(src)), value=self.pad_id)
            padded_sources.append(padded_src)

            # Pad target
            padded_tgt = F.pad(tgt, (0, tgt_max_len - len(tgt)), value=self.pad_id)
            padded_targets.append(padded_tgt)

        # Stack tensors
        src_tensor = torch.stack(padded_sources)
        tgt_tensor = torch.stack(padded_targets)

        decoder_input = tgt_tensor.clone()
        decoder_output = tgt_tensor.clone()

        return {
            "encoder_input": src_tensor.to(self.device),
            "decoder_input": decoder_input.to(self.device),
            "decoder_output": decoder_output.to(self.device),
        }

    def initialize_training(self):
        """Initialize optimizer and learning rate scheduler"""
        # Label smoothing loss
        self.criterion = LabelSmoothingLoss(
            smoothing=self.config.label_smoothing, ignore_index=self.pad_id
        )

        # Adam optimizer with custom betas and epsilon as in the paper
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=0.0,  # Will be set by scheduler
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=float(self.config.adam_epsilon),
        )

        # Noam learning rate scheduler
        self.scheduler = NoamLR(
            optimizer=self.optimizer,
            d_model=self.config.d_model,  # As specified in the paper
            warmup_steps=self.config.warmup_steps,
            factor=self.config.lr_factor,
        )

        # Track best validation loss
        self.best_val_loss = float("inf")
        self.global_step = 0
        self.checkpoint_paths = []
        self.last_checkpoint_time = time.time()

    def train(self):
        """Train the model"""
        logger.info("Starting model training...")
        loss = None
        # Training loop
        while self.global_step < self.config.total_steps:
            self.model.train()

            with tqdm(
                total=len(self.train_loader), desc=f"Step {self.global_step}"
            ) as pbar:
                for batch in self.train_loader:
                    # Forward pass
                    loss = self.train_step(batch)

                    # Check if it's time to save a checkpoint
                    current_time = time.time()
                    time_since_last_checkpoint = (
                        current_time - self.last_checkpoint_time
                    )
                    if (
                        time_since_last_checkpoint
                        >= self.config.checkpoint_interval_minutes * 60
                    ):
                        self.validate_and_checkpoint()
                        self.last_checkpoint_time = current_time

                    # Update progress bar
                    pbar.update(1)
                    pbar.set_postfix(loss=loss.item(), lr=self.scheduler.get_rate())

                    # Check if we've reached total steps
                    if self.global_step >= self.config.total_steps:
                        break

            # After each epoch, validate and save checkpoint
            self.validate_and_checkpoint()
        # After training, average the last N checkpoints
        self.average_checkpoints()

        if self.config.wandb_project_name:
            wandb.finish()

    def train_step(self, batch):
        """
        Perform a single training step.

        Args:
            batch: Dictionary containing encoder input, decoder input, and decoder output

        Returns:
            loss: Training loss for this batch
        """
        # Get batch data
        src = batch["encoder_input"]
        tgt_input = batch["decoder_input"]
        tgt_output = batch["decoder_output"]

        # Create masks
        src_mask = create_padding_mask(src, self.pad_id)
        tgt_mask = create_padding_mask(
            tgt_input, self.pad_id
        ).bool() & create_look_ahead_mask(tgt_input.size(1)).bool().to(self.device)

        # Forward pass
        self.optimizer.zero_grad()
        output, _ = self.model(src, tgt_input, src_mask, tgt_mask)

        # Calculate loss (ignoring padding)
        loss = self.criterion(
            output.contiguous().view(-1, output.size(-1)),
            tgt_output.contiguous().view(-1),
        )

        # Backward pass
        loss.backward()

        # Gradient clipping as mentioned in the paper
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_grad)

        # Update weights and learning rate
        self.optimizer.step()
        self.scheduler.step()
        self.global_step += 1

        # Log to WandB
        if self.config.wandb_project_name:
            wandb.log(
                {
                    "train/loss": loss.item(),
                    "train/learning_rate": self.scheduler.get_rate(),
                    "train/step": self.global_step,
                }
            )

        return loss

    def validate_and_checkpoint(self):
        """Validate the model and save a checkpoint if it's the best so far"""
        val_loss, val_bleu = self.evaluate(self.valid_dataset)
        logger.info(
            f"Step {self.global_step}: Val loss: {val_loss:.4f}, Val BLEU: {val_bleu:.4f}"
        )

        if self.config.wandb_project_name:
            wandb.log(
                {
                    "valid/loss": val_loss,
                    "valid/bleu": val_bleu,
                    "valid/step": self.global_step,
                }
            )

        # Save checkpoint
        checkpoint_path = os.path.join(
            self.config.trained_model_path.parent,
            f"checkpoint_step_{self.global_step}.pt",
        )
        torch.save(self.model.state_dict(), checkpoint_path)

        # Keep track of checkpoint paths for averaging later
        self.checkpoint_paths.append(checkpoint_path)
        if len(self.checkpoint_paths) > self.config.last_n_checkpoints_to_avg:
            self.checkpoint_paths.pop(0)

        # Update best model if this is the best validation loss
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            torch.save(
                self.model.state_dict(),
                os.path.join(self.config.trained_model_path.parent, "best_model.pt"),
            )
            logger.info(f"New best model saved with validation loss: {val_loss:.4f}")

    def evaluate(self, dataset):
        """
        Evaluate the model on a dataset.

        Args:
            dataset: Dataset to evaluate on

        Returns:
            avg_loss: Average loss on the dataset
            bleu: BLEU score
        """
        self.model.eval()
        total_loss = 0

        # Create dataloader for evaluation
        sampler = TokenBucketSampler(
            dataset,
            self.config.src_tokens_per_batch,
            self.config.tgt_tokens_per_batch,
            shuffle=False,
        )
        loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=self.collate_fn)

        # Lists to store generated translations and references for BLEU calculation
        translations = []
        references = []

        with torch.no_grad():
            for batch in loader:
                # Forward pass with teacher forcing for loss calculation
                src = batch["encoder_input"]
                tgt_input = batch["decoder_input"]
                tgt_output = batch["decoder_output"]

                src_mask = create_padding_mask(src, self.pad_id)
                tgt_mask = create_padding_mask(
                    tgt_input, self.pad_id
                ).bool() & create_look_ahead_mask(tgt_input.size(1)).bool().to(
                    self.device
                )

                output, _ = self.model(src, tgt_input, src_mask, tgt_mask)
                loss = self.criterion(
                    output.contiguous().view(-1, output.size(-1)),
                    tgt_output.contiguous().view(-1),
                )
                total_loss += loss.item()

                # Generate translations with beam search for BLEU calculation
                for i in range(src.size(0)):
                    # Extract single source sentence
                    source = src[i : i + 1]
                    source_mask = src_mask[i : i + 1]

                    # Generate translation
                    translation = self.beam_search(
                        source,
                        source_mask,
                        self.config.beam_size,
                        self.config.length_penalty,
                    )

                    # Convert token IDs back to text
                    translation_text = self.tokenizer.decode(
                        translation, skip_special_tokens=True
                    )
                    reference_text = self.tokenizer.decode(
                        tgt_output[i].tolist(), skip_special_tokens=True
                    )

                    translations.append(translation_text)
                    references.append([reference_text])

        # Calculate average loss
        avg_loss = total_loss / len(loader)

        # Calculate BLEU score
        bleu = self.calculate_bleu(references, translations)

        return avg_loss, bleu

    def beam_search(self, src, src_mask, beam_size, length_penalty):
        """
        Perform beam search decoding.

        Args:
            src: Source sequence tensor [1, seq_len]
            src_mask: Source mask tensor
            beam_size: Beam size
            length_penalty: Length penalty factor

        Returns:
            best_hyp: Best hypothesis (list of token IDs)
        """
        # Encode the source sequence
        with torch.no_grad():
            encoder_output, _ = self.model.encoder(src, src_mask)

        # Initialize the first hypothesis with start token
        hyps = [([self.bos_id], 0.0)]
        completed_hyps = []

        max_len = int(src.size(1) * 1.5)  # Heuristic for max target length

        # Beam search
        for step in range(max_len):
            if len(completed_hyps) >= beam_size:
                break

            new_hyps = []

            for hyp, score in hyps:
                if hyp[-1] == self.eos_id:
                    # If hypothesis ends with EOS, add to completed
                    # Apply length penalty: (5 + len(hyp))^length_penalty / (5 + 1)^length_penalty
                    normalized_score = score * (
                        (5 + len(hyp)) ** length_penalty / (5 + 1) ** length_penalty
                    )
                    completed_hyps.append((hyp, normalized_score))
                    continue

                # Convert hypothesis to tensor and create decoder input
                decoder_input = torch.tensor([hyp], device=self.device)

                # Create look-ahead mask for decoder
                tgt_mask = create_look_ahead_mask(len(hyp)).to(self.device)

                # Get model predictions
                with torch.no_grad():
                    decoder_output, _ = self.model.decoder(
                        decoder_input, encoder_output, tgt_mask, src_mask
                    )
                    logits = decoder_output[0, -1]  # Get logits for the last position
                    log_probs = F.log_softmax(logits, dim=-1)

                    # Get top-k predictions
                    topk_log_probs, topk_indices = log_probs.topk(beam_size)

                    for i in range(beam_size):
                        token_id = topk_indices[i].item()
                        log_prob = topk_log_probs[i].item()
                        new_hyp = hyp + [token_id]
                        new_score = score + log_prob
                        new_hyps.append((new_hyp, new_score))

            # Select top-k hypotheses
            hyps = sorted(new_hyps, key=lambda x: x[1], reverse=True)[:beam_size]

        # If no completed hypotheses, select the best incomplete one
        if not completed_hyps:
            best_hyp = hyps[0][0]
        else:
            # Select the completed hypothesis with the highest score
            completed_hyps = sorted(completed_hyps, key=lambda x: x[1], reverse=True)
            best_hyp = completed_hyps[0][0]

        return best_hyp

    def calculate_bleu(self, references, translations):
        """
        Calculate BLEU score.

        Args:
            references: List of reference translations (each with multiple references)
            translations: List of generated translations

        Returns:
            bleu: BLEU score
        """
        try:
            from sacrebleu import corpus_bleu

            # Process references and translations
            processed_refs = [
                [r[0]] for r in references
            ]  # sacrebleu expects list of list of references

            # Calculate BLEU
            bleu = corpus_bleu(translations, processed_refs)
            return bleu.score
        except ImportError:
            logger.warning(
                "sacrebleu not installed. Using a simple BLEU approximation."
            )

            # Simple n-gram precision calculation as fallback
            def ngram_precision(reference, candidate, n):
                ref_ngrams = set(
                    [tuple(reference[i : i + n]) for i in range(len(reference) - n + 1)]
                )
                cand_ngrams = [
                    tuple(candidate[i : i + n]) for i in range(len(candidate) - n + 1)
                ]

                if not cand_ngrams:
                    return 0

                matches = sum(1 for ngram in cand_ngrams if ngram in ref_ngrams)
                return matches / len(cand_ngrams)

            # Split text into words
            ref_tokens = [[r[0].split() for r in refs] for refs in references]
            trans_tokens = [t.split() for t in translations]

            # Calculate precision for n=1,2,3,4
            precisions = []
            for n in range(1, 5):
                precision = 0
                for i in range(len(trans_tokens)):
                    # Find max precision across multiple references
                    max_precision = max(
                        [
                            ngram_precision(ref, trans_tokens[i], n)
                            for ref in ref_tokens[i]
                        ]
                    )
                    precision += max_precision

                precision /= len(trans_tokens)
                precisions.append(precision)

            # Calculate geometric mean of precisions
            if 0 in precisions:
                return 0

            geo_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))

            # Calculate brevity penalty
            ref_lens = [len(ref[0]) for ref in ref_tokens]
            trans_lens = [len(t) for t in trans_tokens]
            bp = math.exp(min(0, 1 - sum(ref_lens) / sum(trans_lens)))

            # Final BLEU
            bleu = bp * geo_mean * 100
            return bleu

    def average_checkpoints(self):
        """Average weights of the last N checkpoints as described in the paper"""
        if len(self.checkpoint_paths) <= 1:
            logger.info("Not enough checkpoints to average")
            return

        logger.info(f"Averaging last {len(self.checkpoint_paths)} checkpoints...")

        # Initialize average with zeros
        avg_state_dict = {}
        for path in self.checkpoint_paths:
            state_dict = torch.load(path, map_location=self.device)

            if not avg_state_dict:
                # First checkpoint, initialize average
                for k, v in state_dict.items():
                    avg_state_dict[k] = v.clone()
            else:
                # Add to running average
                for k, v in state_dict.items():
                    avg_state_dict[k] += v

        # Divide by number of checkpoints
        for k in avg_state_dict.keys():
            avg_state_dict[k] /= len(self.checkpoint_paths)

        # Save averaged model
        self.model.load_state_dict(avg_state_dict)
        torch.jit.script(self.model).save(self.config.trained_model_path)
        logger.info(f"Averaged model saved at {self.config.trained_model_path}")


class ModelTraining:
    def __init__(self, config: ModelTrainingConfig):
        """
        Constructor for ModelTraining class.

        :param config: ModelTrainingConfig object
        :return: None
        """
        self.config = config

    def run(self):
        """
        Main method to train the Transformer model.

        This method handles Training of the model

        :return: None
        """
        # Create trainer instance
        trainer = ModelTrainer(self.config)

        # Start training
        trainer.train()
