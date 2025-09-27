from ml_utils.data.sequence import patch_chat_template, ChatCollateFn, force_llama_chat_template


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
    tokenizer.chat_template = force_llama_chat_template()

    print("\n=== Batch Processing Example ===")
    
    batch_data = [
        {
            "messages": [
                {"role": "system", "content": "You are a math tutor."},
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hi there! How can I help you today?"}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What's 2+2?"},
                {"role": "assistant", "content": "2+2 equals 4."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a math tutor."},
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
    
    for i in range(len(batch_result['text'])):
        print(batch_result['text'][i])
        for (input_ids, attn, assist) in zip(batch_result['input_ids'][i], batch_result['attention_mask'][i], batch_result['loss_mask'][i]):
            print(input_ids.item(), attn.item(), assist.item())

    
        


if __name__ == "__main__":
    # Run the example
    example_chat_training()