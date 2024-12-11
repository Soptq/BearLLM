import torch
from dotenv import dotenv_values
from peft import PeftModel
from transformers import AutoTokenizer
from fine_tuning import description_len, signal_token_id, get_bearllm, mod_xt_for_qwen
import numpy as np
from BearLLM.functions.dcn import dcn

query_signal_file = 'xxx.npy'  # Replace with the path of the vibration signal file
reference_signal_file = 'xxx.npy'  # Replace with the path of the vibration signal file
user_prompt = 'xxx'  # Replace with the user prompt


def create_cache():
    query_data = np.load(query_signal_file)
    ref_data = np.load(reference_signal_file)
    query_data = dcn(query_data)
    ref_data = dcn(ref_data)
    res_data = query_data - ref_data
    rv = np.array([query_data, ref_data, res_data])
    np.save('./cache.npy', rv)


def run_demo():
    create_cache()

    place_holder_ids = torch.ones(description_len) * signal_token_id
    text_part1, text_part2 = mod_xt_for_qwen(user_prompt)

    tokenizer = AutoTokenizer.from_pretrained(dotenv_values()["qwen_weights_dir"])
    tokenizer.pad_token_id = tokenizer.eos_token_id
    user_part1_ids = tokenizer(text_part1, return_tensors='pt', add_special_tokens=False).input_ids[0]
    user_part2_ids = tokenizer(text_part2, return_tensors='pt', add_special_tokens=False).input_ids[0]
    user_ids = torch.cat([user_part1_ids, place_holder_ids, user_part2_ids])
    attention_mask = torch.ones_like(user_ids)

    model = get_bearllm()
    model = PeftModel.from_pretrained(model, dotenv_values()["lora_weights_dir"])
    model.eval()

    output = model.generate(user_ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0), max_new_tokens=512)
    output_text = tokenizer.decode(output[0, user_ids.shape[0]:], skip_special_tokens=True)
    print(output_text)


if __name__ == "__main__":
    run_demo()
