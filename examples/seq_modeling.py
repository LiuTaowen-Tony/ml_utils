from ml_utils.sequence_modelling import patch_chat_template, ChatCollateFn, prepare_chat_messages_for_training


def example_chat_training():
    """
    Example demonstrating how to use chat message preparation and collation
    for conversational model training.
    """
    from transformers import AutoTokenizer
    
    # Initialize tokenizer (use any chat model tokenizer)
    tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B")

    chat_template = AutoTokenizer.from_pretrained("unsloth/Llama-3.1-8B-Instruct").chat_template
    print(tokenizer.chat_template)
    chat_template = patch_chat_template(chat_template)
    tokenizer.chat_template = chat_template

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
    
    print(batch_result['text'][0])
    print(batch_result['input_ids'][0])
    print(batch_result['labels'][0])
    print(batch_result['attention_mask'][0])

    
        


if __name__ == "__main__":
    # Run the example
    example_chat_training()