from typing import List, Dict, Tuple

from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import json

print("Loading GPT-2 tokenizer...")
tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
print("Tokenizer loaded.")

print("Loading GPT-2 model...")
model: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained('gpt2', pad_token_id=tokenizer.eos_token_id)
model.eval().cuda()
print("Model loaded and moved to GPU.")

def generate(prompt: str, max_length: int = 5, stop_token: str = None) -> str:
    input_ids: torch.Tensor = tokenizer.encode(prompt, return_tensors="pt")
    generated_text_ids: torch.Tensor = model.generate(input_ids=input_ids.cuda(), max_length=max_length+len(input_ids[0]), do_sample=False)
    generated_text: str = tokenizer.decode(generated_text_ids[0], clean_up_tokenization_spaces=True)
    post_prompt_text: str = generated_text[len(tokenizer.decode(input_ids[0], clean_up_tokenization_spaces=True)):]
    return prompt + post_prompt_text[:post_prompt_text.find(stop_token) if stop_token else None]

def get_logits_and_tokens(text: str) -> Tuple[torch.Tensor, List[str]]:
    input_ids: torch.Tensor = tokenizer.encode(text, return_tensors="pt")
    tokens: List[str] = [tokenizer.decode([input_id]) for input_id in input_ids[0]]
    output: torch.Tensor = model(input_ids.cuda())
    return output.logits[0][:-1], tokens

def get_classification(prompt: str, classes: List[str], print_all_probs: bool = False) -> Dict[str, float]:
    if any([not class_name.startswith(" ") for class_name in classes]):
        # This is so I don't mess up during development. This should probably be a warning, or it could automatically add the space, but for now I want to be explicit about what I'm doing.
        raise ValueError("All class names must start with a space.")
    generated_text: str = generate(prompt, stop_token="\n")
    if print_all_probs:
        print(f"Generated Text: `{generated_text}`")
    logits, tokens = get_logits_and_tokens(generated_text)
    last_token_probs: torch.Tensor = torch.softmax(logits[-1], dim=0)
    class_probs: Dict[str, float] = {class_name: last_token_probs[tokenizer.encode(class_name)[0]].item() for class_name in classes}
    
    if print_all_probs:
        all_probs: Dict[str, float] = {tokenizer.decode([i]): prob.item() for i, prob in enumerate(last_token_probs)}
        sorted_probs = sorted(all_probs.items(), key=lambda item: item[1], reverse=True)[:10]
        print("Top 10 token probabilities:", sorted_probs)

    return class_probs

if __name__ == "__main__":
    EXAMPLE_PROMPT: str = """Horrible: negative
    Great: positive
    Bad:"""

    generated_text: str = generate(EXAMPLE_PROMPT, stop_token="\n")
    print(f"Generated Text: {generated_text}")

    logits, tokens = get_logits_and_tokens(generated_text)
    last_token_probs: torch.Tensor = torch.softmax(logits[-1], dim=0)
    negative_prob: torch.Tensor = last_token_probs[tokenizer.encode(" negative")[0]]
    positive_prob: torch.Tensor = last_token_probs[tokenizer.encode(" positive")[0]]

    print(f"tokens: {tokens}\nnegative prob: {negative_prob}\npositive prob: {positive_prob}")

    # Use the generated text for classification
    classification_result = get_classification(EXAMPLE_PROMPT, [" negative", " positive"])
    print(f"Classification Result: {classification_result}")
