import torch
from transformers import PreTrainedTokenizerFast
import typing
from ml_utils.dist import rank0_print

#### utils

def force_llama_chat_template():
    return """{%- set bos = bos_token | default('<|begin_of_text|>') -%}
{%- set eot = '<|eot_id|>' -%}
{%- set sh  = '<|start_header_id|>' -%}
{%- set eh  = '<|end_header_id|>' -%}

{{ bos }}
{%- for message in messages -%}
{{ sh }}{{ message['role'] }}{{ eh }}

{%- if message['role'] == 'assistant' -%}
{% generation %}{{ message['content'] | trim }}{{ eot }}{% endgeneration %}
{%- else -%}
{{ message['content'] | trim }}{{ eot }}
{%- endif -%}
{%- endfor -%}

{%- if add_generation_prompt -%}
{{ sh }}assistant{{ eh }}

{%- endif -%}
"""




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

#### collator

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


class ChatCollateFn:
    """
    Collate function for conversational training data that properly masks
    user/system messages so only assistant responses contribute to loss.
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        max_seq_len: int,
        padding: typing.Literal["longest", "max_length"] = "longest",
        messages_key: str = "messages",
        pad_to_multiple_of: int = 16,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
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
        
        # Create labels: mask non-assistant tokens with ignore_index
        # labels = torch.full_like(input_ids, self.ignore_index)
        # labels[:, :-1] = input_ids[:, 1:]
        # labels[~assistant_mask] = self.ignore_index
        
        # Create attention mask (all tokens are attended to)
        str_messages = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        
        return {
            'text': str_messages,
            'input_ids': input_ids,
            'loss_mask': result['assistant_masks'].bool(),
            'attention_mask': result['attention_mask'],
        }

### batch_map

class TokenizeThenGroupBatchMap:
    """
    Batch map function that tokenizes the text and groups the input_ids into chunks of max_seq_len.
    usage:
    dataset = dataset.map(TokenizeThenGroupBatchMap(tokenizer, max_seq_len), batched=True, num_proc=32, batch_size=200, remove_columns=dataset.column_names)
    remove columns is important
    """

    def __init__(self, tokenizer: PreTrainedTokenizerFast, max_seq_len: int):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __call__(self, examples):
        text_joined = "\n\n".join(examples["text"])
        encoded = self.tokenizer(text_joined)
        input_ids = encoded["input_ids"]
        total_length = len(input_ids)
        if total_length >= self.max_seq_len:
            total_length = (total_length // self.max_seq_len) * self.max_seq_len
        list_of_input_ids = [input_ids[i : i + self.max_seq_len] for i in range(0, total_length, self.max_seq_len)]
        return {
            "input_ids": list_of_input_ids,
        }



