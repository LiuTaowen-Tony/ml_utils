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
    """
    Prepare conversational messages for training by applying chat template and masking 
    user/system prompts so only assistant responses contribute to loss.
    
    Args:
        messages: List of message dicts with 'role' and 'content' keys
        tokenizer: HuggingFace tokenizer with chat template support
        max_length: Maximum sequence length for truncation
        ignore_index: Index to use for masked tokens (default: -100)
    
    Returns:
        Dict containing:
        - input_ids: Tokenized conversation
        - labels: input_ids with user/system tokens masked with ignore_index
        - attention_mask: Attention mask for the sequence
        
    Example:
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        result = prepare_chat_messages_for_training(messages, tokenizer)
    """
    result = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_assistant_tokens_mask=True,
        return_dict=True,
        max_length=max_length,
        truncation=True if max_length else False,
        add_generation_prompt=False
    )
    
    input_ids = torch.tensor(result['input_ids'])
    assistant_mask = torch.tensor(result['assistant_masks'], dtype=torch.bool)
    
    # Create labels: mask non-assistant tokens with ignore_index
    labels = input_ids.clone()
    labels[~assistant_mask] = ignore_index
    
    # Create attention mask (all tokens are attended to)
    attention_mask = torch.ones_like(input_ids)
    
    return {
        'input_ids': input_ids,
        'labels': labels,
        'attention_mask': attention_mask,
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
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.ignore_index = ignore_index
        self.padding = padding
        self.messages_key = messages_key
    
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
        )
        input_ids = result['input_ids']
        assistant_mask = result['assistant_masks'].bool()
        
        # Create labels: mask non-assistant tokens with ignore_index
        labels = input_ids.clone()
        labels[~assistant_mask] = self.ignore_index
        
        # Create attention mask (all tokens are attended to)
        attention_mask = torch.ones_like(input_ids)
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
        }


# DataCollator takes list of dicts
# Batch map in datasets takes list of dicts


def example_chat_training():
    """
    Example demonstrating how to use chat message preparation and collation
    for conversational model training.
    """
    from transformers import AutoTokenizer
    
    # Initialize tokenizer (use any chat model tokenizer)
    tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.1-8B-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(tokenizer.chat_template)
    
    # Update chat template to support assistant token masking
    # Replace the main message rendering line with generation-aware version
    original_template = tokenizer.chat_template
    
    # Find and replace the main message rendering line
    old_line = "{{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'+ message['content'] | trim + '<|eot_id|>' }}"
    new_line = """{%- if message['role'] == 'assistant' -%}
        {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}{% generation %}{{ message['content'] | trim }}{% endgeneration %}{{- '<|eot_id|>' }}
    {%- else -%}
        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'+ message['content'] | trim + '<|eot_id|>' }}
    {%- endif -%}"""
    
    modified_template = original_template.replace(old_line, new_line)
    
    tokenizer.chat_template = modified_template
    
    print("=== Chat Message Preparation Example ===")
    
    # Example conversation
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
        {"role": "user", "content": "What about Germany?"},
        {"role": "assistant", "content": "The capital of Germany is Berlin."}
    ]
    
    print("Original messages:")
    for msg in messages:
        print(f"  {msg['role']}: {msg['content']}")
    
    result = prepare_chat_messages_for_training(messages, tokenizer, max_length=128)
    
    print(f"\nProcessed conversation:")
    print(f"  Input IDs shape: {result['input_ids'].shape}")
    print(f"  Labels shape: {result['labels'].shape}")
    print(f"  Attention mask shape: {result['attention_mask'].shape}")
    
    # Show which tokens are masked (labels == -100)
    masked_positions = (result['labels'] == -100).sum().item()
    total_tokens = len(result['labels'])
    print(f"  Masked tokens (user/system): {masked_positions}/{total_tokens}")
    print(f"  Training tokens (assistant): {total_tokens - masked_positions}/{total_tokens}")
    
    # Decode to show the formatted conversation
    decoded_text = tokenizer.decode(result['input_ids'], skip_special_tokens=False)
    print(f"\nFormatted conversation text:")
    print(f"  {repr(decoded_text[:200])}")
    
    print("\n=== Batch Processing Example ===")
    
    batch_data = [
        {
            "messages": [
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hi there! How can I help you today?"}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a math tutor."},
                {"role": "user", "content": "What's 2+2?"},
                {"role": "assistant", "content": "2+2 equals 4."}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Tell me a joke."},
                {"role": "assistant", "content": "Why don't scientists trust atoms? Because they make up everything!"}
            ]
        }
    ]
    
    print(f"Processing batch of {len(batch_data)} conversations...")
    
    # Initialize collate function
    collate_fn = ChatCollateFn(
        tokenizer=tokenizer,
        max_seq_len=128,
        ignore_index=-100,
        padding="longest"
    )
    
    # Process batch
    batch_result = collate_fn(batch_data)
    
    print(batch_result['input_ids'][0])
    print(batch_result['labels'][0])
    print(batch_result['attention_mask'][0])

    
        


if __name__ == "__main__":
    # Run the example
    example_chat_training()