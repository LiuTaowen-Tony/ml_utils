import typing
import torch
from transformers import PreTrainedTokenizerFast
from ml_utils.dist import rank0_print

def boolean_triangular_mask(flat_mask: torch.Tensor) -> torch.BoolTensor:
    assert flat_mask.dim() < 3, "Input tensor should be 1D or 2D"
    if flat_mask.dim() == 1:
        flat_mask = flat_mask.unsqueeze(0)
    flat_mask = flat_mask.bool()

    batch_size, seq_len = flat_mask.size()

    # external product of flat_mask with itself
    tri_mask = flat_mask.unsqueeze(2) & flat_mask.unsqueeze(1)

    # upper triangular mask
    upper_mask = torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=flat_mask.device),
        diagonal=1,
    )
    lower_mask = upper_mask.logical_not()

    tri_mask = tri_mask & lower_mask
    return tri_mask

def find_token_sequence(input_ids: torch.Tensor, token_sequence: torch.Tensor):
    """
    Find the starting indices of a specific token sequence in input_ids

    :param input_ids: Input token ids tensor
    :param token_sequence: Sequence of tokens to find
    :return: Tensor of starting indices
    """
    assert isinstance(input_ids, torch.Tensor), "Input_ids should be a tensor"
    assert isinstance(token_sequence, torch.Tensor), "Token_sequence should be a tensor"
    assert input_ids.dim() == 1, "Input_ids should be a 1D tensor"
    assert token_sequence.dim() == 1, "Token_sequence should be a 1D tensor"

    seq_len = len(token_sequence)
    matches = []

    for i in range(len(input_ids) - seq_len + 1):
        if torch.equal(input_ids[i : i + seq_len], token_sequence):
            matches.append(i)

    return torch.tensor(matches, dtype=torch.long)


def mask_between_tokens(input_ids, start_tokens, end_tokens):
    """
    Mask the tokens between the start and end tokens

    :param input_ids: Input token ids tensor
    :param start_tokens: Start tokens
    :param end_tokens: End tokens
    :return: Masked input_ids
    """
    assert input_ids.dim() == 1
    labels = input_ids.clone()

    # Find system response start positions
    system_start_positions = find_token_sequence(input_ids, start_tokens)
    eos_positions = find_token_sequence(input_ids, end_tokens)
    # Process each system response
    for start_pos in system_start_positions:
        # Find the end of this system response
        end_pos = start_pos + len(start_tokens)

        # Look for next user start or end of sequence
        next_end_pos = min(
            eos_positions[eos_positions > end_pos], default=len(input_ids)
        )
        if next_end_pos == len(input_ids):
            next_end_pos = len(input_ids) - 1
        labels[start_pos:next_end_pos] = -100

    assert input_ids.shape == labels.shape
    return labels


def apply_next_token_shift(
    input_ids,
):
    """
    Apply a next token shift to the input_ids

    :param input_ids: Input token ids tensor
    :return: Shifted input_ids
    """

    labels = torch.full_like(input_ids, -100)
    if input_ids.dim() == 1:
        labels[:-1] = input_ids[1:]
    elif input_ids.dim() == 2:
        labels[:, :-1] = input_ids[:, 1:]
    else:
        raise ValueError(
            f"Input_ids should be a 1D or 2D tensor, got {input_ids.dim()}D"
        )
    return labels




def prepare_chat_messages_for_training(
    messages: list[dict],
    tokenizer: PreTrainedTokenizerFast,
    max_length: int = None,
    ignore_index: int = -100,
) -> dict:
    collator_fn = ChatCollateFn(tokenizer, max_length, ignore_index)
    result = collator_fn([{"messages": messages}])
    return {
        "input_ids": result["input_ids"][0],
        "labels": result["labels"][0],
        "attention_mask": result["attention_mask"][0],
        "text": result["text"][0],
    }
        

# collator : takes list of dicts, return dict of lists

