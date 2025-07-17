import typing
import torch
from transformers import PreTrainedTokenizerFast


def text_token_numbers(tokenizer: PreTrainedTokenizerFast, text):
    # Tokenize the text and count the number of tokens
    return tokenizer.encode(text, return_tensors="pt").numel()


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

    # # tri_mask = torch.triu(torch.ones(batch_size, seq_len, seq_len, dtype=torch.bool), diagonal=1)
    # # tri_mask = tri_mask.logical_not()

    # return flat_mask.unsqueeze(1) & tri_mask


def pad_to_length(
    tensor: torch.Tensor,
    length: int,
    pad_value: typing.Union[int, float],
    dim: int = -1,
) -> torch.Tensor:
    """Pad a tensor to a specific length along a specific dimension"""
    if tensor.size(dim) >= length:
        return tensor
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat(
            [
                tensor,
                pad_value
                * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device),
            ],
            dim=dim,
        )


def extract_anthropic_prompt(prompt_and_response):
    """Extract the anthropic prompt from a prompt and response pair."""
    search_term = "\n\nAssistant:"
    search_term_idx = prompt_and_response.rfind(search_term)
    assert (
        search_term_idx != -1
    ), f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[: search_term_idx + len(search_term)]


def strip_html_tags(html_string):
    from bs4 import BeautifulSoup, NavigableString

    """Strip HTML tags from a string, except for <code> tags (which contain real code in the StackExchange answers)."""
    # Create a BeautifulSoup object
    soup = BeautifulSoup(html_string, "html.parser")

    # Initialize an empty list to store the text
    text = []
    for element in soup.children:
        if isinstance(element, NavigableString):
            continue
        if element.name == "p":
            text.append(
                "".join(
                    child.string
                    for child in element.children
                    if isinstance(child, NavigableString)
                )
            )
        elif element.name == "pre":
            for code in element.find_all("code"):
                text.append("<code>" + code.get_text() + "</code>")
        elif element.name == "code":
            text.append("<code>" + element.get_text() + "</code>")

    # Join the text together with newlines in between
    text = "\n\n".join(text)

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


def tokenizing_collate_fn(
    tokenizer: PreTrainedTokenizerFast, max_seq_len: int, batch: list[dict]
):
    strs = [x["text"] for x in batch]
    encoding = tokenizer(
        strs,
        truncation=True,
        padding="max_length",
        max_length=max_seq_len,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"]

    # Shift labels by one position to the right
    labels = input_ids.clone()
    labels = torch.cat([labels[:, 1:], labels[:, :1]], dim=1)  # Shift left
    labels[labels == tokenizer.pad_token_id] = -100

    token_count = (input_ids != tokenizer.pad_token_id).sum().item()
    return {
        "input_ids": input_ids,
        "labels": labels,
        "token_count": token_count,
        "attention_mask": encoding["attention_mask"],
        "text": strs,
    }
