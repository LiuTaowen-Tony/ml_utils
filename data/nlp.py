import inspect
import functools
import torch
from transformers import PreTrainedTokenizerFast


def text_token_numbers(tokenizer: PreTrainedTokenizerFast, text):
    # Tokenize the text and count the number of tokens
    return tokenizer.encode(text, return_tensors="pt").numel()


def apply_chat_template(messages):
    # Apply a chat template to messages
    text = ""
    for message in messages:
        if message["role"] == "user":
            text += f"<|user|>{message['content']}</s>"
        elif message["role"] == "assistant":
            text += f"<|assistant|>{message['content']}</s>"
    return text


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


def tokenize_encode_pad_max_len(tokenizer: PreTrainedTokenizerFast, max_seq_len, text):
    """
    Tokenize and encode a text string

    :param tokenizer: Tokenizer object
    :param text: Input text
    :return: input_ids and attention_mask
    """
    # Tokenize and encode the text
    encoding = tokenizer.encode_plus(
        text,
        padding="max_length",
        max_length=max_seq_len,
        truncation=True,
        return_tensors="pt",
    )

    return {"input_ids": encoding["input_ids"], "attn_mask": encoding["attention_mask"]}


def labels_skip_user_prompts(
    system_start_tokens,
    eos_token,
    input_ids,
):
    """
    Process input messages to create input_ids and labels for training

    :param item: Dictionary containing 'messages' list
    :return: Dictionary with input_ids and labels
    """
    # Initialize labels with -100 (ignore tokens)
    assert input_ids.dim() == 2

    input_ids = input_ids[0]
    labels = torch.full_like(input_ids, -100)

    # Find system response start positions
    system_start_positions = find_token_sequence(input_ids, system_start_tokens)
    eos_positions = find_token_sequence(input_ids, eos_token)

    # Process each system response
    for start_pos in system_start_positions:
        # Find the end of this system response
        end_pos = start_pos + len(system_start_tokens)

        # Look for next user start or end of sequence
        next_end_pos = min(
            eos_positions[eos_positions > end_pos], default=len(input_ids)
        )
        if next_end_pos == len(input_ids):
            next_end_pos = len(input_ids) - 1
        labels[end_pos - 1 : next_end_pos - 1] = input_ids[end_pos:next_end_pos]

    assert input_ids.shape == labels.shape
    return labels