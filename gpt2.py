from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import json

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained('gpt2', pad_token_id=tokenizer.eos_token_id)
model.eval().cuda()

def generate(prompt, max_length=5, stop_token=None):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    generated_text_ids = model.generate(input_ids=input_ids.cuda(), max_length=max_length+len(input_ids[0]), do_sample=False)
    generated_text = tokenizer.decode(generated_text_ids[0], clean_up_tokenization_spaces=True)
    post_prompt_text = generated_text[len(tokenizer.decode(input_ids[0], clean_up_tokenization_spaces=True)):]
    return prompt + post_prompt_text[:post_prompt_text.find(stop_token) if stop_token else None]

# Note that the logits are shifted over 1 to the left, since HuggingFace doesn't give a logit for the first token
def get_logits_and_tokens(text):
    input_ids = tokenizer.encode(text, return_tensors="pt")
    tokens = [tokenizer.decode([input_id]) for input_id in input_ids[0]]
    output = model(input_ids.cuda())
    return output.logits[0][:-1], tokens

if __name__ == "__main__":
    EXAMPLE_PROMPT = """Horrible: negative
    Great: positive
    Bad:"""

    generated_text = generate(EXAMPLE_PROMPT, stop_token="\n")
    print(f"Generated Text: {generated_text}")

    logits, tokens = get_logits_and_tokens(generated_text)
    last_token_probs = torch.softmax(logits[-1], dim=0)
    negative_prob = last_token_probs[tokenizer.encode(" negative")[0]]
    positive_prob = last_token_probs[tokenizer.encode(" positive")[0]]

    print(f"tokens: {tokens}\nnegative prob: {negative_prob}\npositive prob: {positive_prob}")