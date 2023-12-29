from dataclasses import dataclass
from typing import Tuple, Optional, Union, List

import numpy as np
import torch
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import Seq2SeqLMOutput
import evaluate

perplexity_metric = evaluate.load("perplexity")
accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")


def debug_data_processing(train_dataloader):
    batch = None
    for batch in train_dataloader:
        break
    print({k: v.shape for k, v in batch.items()})
    return batch


def freeze_original_bart_params(model):
    for n, p in model.named_parameters():
        if "encoder" in n or "decoder" in n:
            if "decoder.embed_positions.weight" in n:
                continue
            p.requires_grad = False

def freeze_expert_params(model, expert_idx):
    for n, p in model.named_parameters():
        if f"experts.{expert_idx}" in n:
            p.requires_grad = False



def _prepare_4d_causal_attention_mask(
    attention_mask: Optional[torch.Tensor],
    input_shape: Union[torch.Size, Tuple, List],
    inputs_embeds: torch.Tensor,
    past_key_values_length: int,
    sliding_window: Optional[int] = 512,
):
    """
    Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`

    Args:
        attention_mask (`torch.Tensor` or `None`):
            A 2D attention mask of shape `(batch_size, key_value_length)`
        input_shape (`tuple(int)` or `list(int)` or `torch.Size`):
            The input shape should be a tuple that defines `(batch_size, query_length)`.
        inputs_embeds (`torch.Tensor`):
            The embedded inputs as a torch Tensor.
        past_key_values_length (`int`):
            The length of the key value cache.
        sliding_window (`int`, *optional*):
            If the model uses windowed attention, a sliding window should be passed.
    """
    attn_mask_converter = AttentionMaskConverter(is_causal=True, sliding_window=sliding_window)

    key_value_length = input_shape[-1] + past_key_values_length

    # 4d mask is passed through the layers
    if attention_mask is not None:
        attention_mask = attn_mask_converter.to_4d(
            attention_mask, input_shape[-1], key_value_length, dtype=inputs_embeds.dtype
        )
    else:
        attention_mask = attn_mask_converter.to_causal_4d(
            input_shape[0], input_shape[-1], key_value_length, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )

    return attention_mask


def evaluate(model, eval_dataloader, accelerator):
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(batch["input_ids"], labels=batch["input_ids"], expert_train=batch["expert_train"])

        losses.append(accelerator.gather(outputs.loss))
    loss = torch.mean(torch.stack(losses))
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = float("inf")
    return loss.item(), perplexity.item()


def evaluate_classification(model, eval_dataloader, accelerator):
    model.eval()
    losses = []
    correct_predictions = 0
    total_predictions = 0

    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            inputs = batch["input_ids"]
            labels = batch["encoder_labels"]
            outputs = model(inputs)

            loss = torch.nn.functional.cross_entropy(outputs.encoder_logits, labels)
            losses.append(accelerator.gather(loss))

            preds = torch.argmax(outputs.encoder_logits, dim=1)
            correct_predictions += (preds == labels).sum().item()
            total_predictions += labels.shape[0]

    avg_loss = torch.mean(torch.stack(losses)).item()
    accuracy = correct_predictions / total_predictions

    return avg_loss, accuracy


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    precisions = []
    accuracies = []
    predictions = []
    for b_logits in logits:
        padding_size = len(labels[0]) - len(b_logits)
        if padding_size > 0:
            b_logits = np.pad(b_logits, (0, padding_size), 'constant', constant_values=-100)
        predictions.append(b_logits)
    for i in range(len(logits)):
        precisions.append(precision_metric.compute(predictions=predictions[i], references=labels[i], average='macro')['precision'])
        accuracies.append(accuracy_metric.compute(predictions=predictions[i], references=labels[i], )['accuracy'])
    return {
        "accuracy": np.mean(accuracies),
        "precision": np.mean(precisions),
    }



@dataclass
class Seq2SeqMoeModelOutput(Seq2SeqLMOutput):
    encoder_loss: Optional[torch.FloatTensor] = None
    encoder_logits: torch.FloatTensor = None
