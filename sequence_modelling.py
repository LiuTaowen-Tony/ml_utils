import typing
import torch
from transformers import PreTrainedTokenizerFast
from ml_utils.dist import rank0_print


def patch_chat_template(chat_template: str):
    original_template = chat_template
    if original_template is None:
        raise ValueError("Chat template is None")

    is_llama = "'<|start_header_id|>'" in original_template
    is_deepseek = "'<｜Assistant｜>'" in original_template
    is_qwen = "'<|im_start|>'" in original_template
    rank0_print(f"Template type - Llama: {is_llama}, DeepSeek: {is_deepseek}, Qwen: {is_qwen}")

    llama_old = "{{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'+ message['content'] | trim + '<|eot_id|>' }}"
    llama_new= """{%- if message['role'] == 'assistant' -%}
    {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}{% generation %}{{ message['content'] | trim }}{% endgeneration %}{{- '<|eot_id|>' }}
{%- else -%}
    {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'+ message['content'] | trim + '<|eot_id|>' }}
{%- endif -%}"""
    deepseek_old = "{{'<｜Assistant｜>' + content + '<｜end▁of▁sentence｜>'}}"
    deepseek_new = "{{'<｜Assistant｜>'}}{% generation %}{{ content }}{% endgeneration %}{{'<｜end▁of▁sentence｜>'}}"
    
    
    if llama_old in original_template:
        modified_template = original_template.replace(llama_old, llama_new)
        rank0_print(f"Llama replacement found: {llama_old in original_template}")
    elif deepseek_old in original_template:
        modified_template = original_template.replace(deepseek_old, deepseek_new)
        rank0_print(f"DeepSeek replacement found: {deepseek_old in original_template}")
    else:
        raise ValueError("Unknown template format")
    return modified_template

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


class TokenizingCollateFn:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        max_seq_len: int,
        padding: typing.Literal["longest", "max_length"] = "longest",
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.padding = padding

    def __call__(self, batch: list[dict]):
        strs = [x["text"] for x in batch]
        encoding = self.tokenizer(
            strs,
            truncation=True,
            padding=self.padding,
            max_length=self.max_seq_len,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"]

        # Shift labels by one position to the right
        labels = torch.full_like(input_ids, -100)
        labels[:, :-1] = input_ids[:, 1:]
        labels[labels == self.tokenizer.pad_token_id] = -100

        token_count = (input_ids != self.tokenizer.pad_token_id).sum().item()
        result =  {
            "input_ids": input_ids,
            "labels": labels,
            "token_count": token_count,
            "attention_mask": encoding["attention_mask"],
            "text": strs,
        }

        return result


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
        

class ChatCollateFn:
    """
    Collate function for conversational training data that properly masks
    user/system messages so only assistant responses contribute to loss.
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        max_seq_len: int,
        ignore_index: int = -100,
        padding: typing.Literal["longest", "max_length"] = "longest",
        messages_key: str = "messages",
        pad_to_multiple_of: int = 16,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.ignore_index = ignore_index
        self.padding = padding
        self.messages_key = messages_key
        self.pad_to_multiple_of = pad_to_multiple_of
    
    def __call__(self, batch: list[dict]) -> dict:
        """
        Process a batch of conversations.
        
        Args:
            batch: List of dicts, each containing 'messages' key with conversation
        
        Returns:
            Batched tensors ready for training
        """
        messages = [x[self.messages_key] for x in batch]
        result = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_assistant_tokens_mask=True,
            return_dict=True,
            max_length=self.max_seq_len,
            padding=self.padding,
            truncation=True if self.max_seq_len else False,
            add_generation_prompt=False,
            return_tensors="pt",
            tokenizer_kwargs={ "pad_to_multiple_of": self.pad_to_multiple_of }
        )
        input_ids = result['input_ids']
        assistant_mask = result['assistant_masks'].bool()
        
        # Create labels: mask non-assistant tokens with ignore_index
        labels = torch.full_like(input_ids, self.ignore_index)
        labels[:, :-1] = input_ids[:, 1:]
        labels[~assistant_mask] = self.ignore_index
        
        # Create attention mask (all tokens are attended to)
        str_messages = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        
        return {
            'text': str_messages,
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': result['attention_mask'],
        }


# DataCollator takes list of dicts
# Batch map in datasets takes list of dicts

