import os
import torch
from dotenv import dotenv_values
from peft import LoraConfig, TaskType, get_peft_model
from torch import nn
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, TrainingArguments, Trainer
import numpy as np
from BearLLM.models.FCN import FeatureEncoder
from BearLLM.functions.mbhm import MBHMDataset, MBHMVibrationDataset


class HyperParameters:
    def __init__(self):
        self.device = 'cuda'
        self.r = 4  # 低秩分解的秩
        self.lora_alpha = 32  # LoRA的alpha参数
        self.lora_dropout = 0.1  # Dropout比例
        self.per_device_train_batch_size = 4
        self.gradient_accumulation_steps = 4
        self.logging_steps = 10
        self.num_train_epochs = 1
        self.save_steps = 100
        self.learning_rate = 1e-4
        self.lr_scheduler_type = 'cosine'


description_text = [
    "normal bearing condition",
    "minor fault of bearing inner ring",
    "moderate fault of bearing inner ring",
    "severe fault of bearing inner ring",
    "minor fault of bearing ball",
    "moderate fault of bearing ball",
    "severe fault of bearing ball",
    "minor fault of bearing outer ring",
    "moderate fault of bearing outer ring",
    "severe fault of bearing outer ring",
]
description_len = 7
llm_hidden_size = 1536
signal_token_id = 151925

sys_prompt = ("As an expert in bearing fault diagnosis with extensive knowledge in mechanical engineering and failure "
              "analysis, you can assess the condition of bearings. Typically, bearing states are categorized as "
              "normal, outer ring fault, inner ring fault, and ball fault. These defects are further classified into "
              "three levels: minor, moderate, and severe. Based on your description of the bearing state, "
              "you will answer my questions concisely and directly, providing only the answer without reiterating the "
              "user's prompt or bearing status description.")


def initialize_l3_weight():
    hp = HyperParameters()
    tokenizer = AutoTokenizer.from_pretrained(dotenv_values()['QWEN_WEIGHTS_DIR'],
                                              device_map=hp.device,
                                              torch_dtype="auto",
                                              trust_remote_code=True)
    tokenizer.pad_token_id = 0
    token_ids = tokenizer(description_text, padding=True, add_special_tokens=False).input_ids
    token_ids = torch.tensor(token_ids).to(hp.device)

    llm = AutoModelForCausalLM.from_pretrained(dotenv_values()['QWEN_WEIGHTS_DIR'])
    embedding = llm.get_input_embeddings().to(hp.device)
    embeds = embedding(token_ids).to(torch.float32).detach().cpu().numpy()
    os.makedirs(dotenv_values()['ALIGN_WEIGHTS_DIR'], exist_ok=True)
    np.save(f'{dotenv_values()['ALIGN_WEIGHTS_DIR']}/l3.npy', embeds)
    return embeds


def load_l3_weight():
    if not os.path.exists(f'{dotenv_values()['ALIGN_WEIGHTS_DIR']}/l3.npy'):
        return initialize_l3_weight()
    return np.load(f'{dotenv_values()['ALIGN_WEIGHTS_DIR']}/l3.npy')


class IdConverter:
    def __init__(self):
        self.dataset = MBHMVibrationDataset()
        self.hp = HyperParameters()
        self.test_file = './cache.npy'

    def get_signal(self, signal_ids_tensor, is_train=True):
        res = []
        if is_train:
            for i in range(0, len(signal_ids_tensor), description_len):
                signal_id = signal_ids_tensor[i] - signal_token_id
                rv, _ = self.dataset.__getitem__(signal_id)
                res.append(rv)
        else:
            res.append(np.load(self.test_file))
            os.remove(self.test_file)
        data = np.array(res)
        return torch.Tensor(data).to(self.hp.device).detach()


class AlignmentLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(128 * 47, 128)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(128, 10)
        self.softmax = nn.Softmax(dim=1)
        self.linear3 = nn.Linear(10, llm_hidden_size)

    def forward(self, x):
        x = x.view(-1, 47 * 128)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.softmax(x)
        x = self.linear3(x)
        x = x.reshape(x.size(0), -1, llm_hidden_size)
        x = x.to(torch.bfloat16)
        return x

    def load_default(self):
        classifier_weights = torch.load(f'{dotenv_values()['FCN_WEIGHTS_DIR']}/classifier.pth', map_location='cpu')
        self.linear1.weight.data = classifier_weights['linear1.weight']
        self.linear1.bias.data = classifier_weights['linear1.bias']
        self.linear2.weight.data = classifier_weights['linear2.weight']
        self.linear2.bias.data = classifier_weights['linear2.bias']
        l3_weight = load_l3_weight()
        l3_weight = torch.from_numpy(l3_weight)
        l3_weight = l3_weight.reshape(l3_weight.size(0), -1)
        self.linear3.weight.data = l3_weight.T
        self.linear3.bias.data = torch.zeros(l3_weight.size(1))

    def save_weights(self):
        torch.save(self.state_dict(), dotenv_values()['ALIGN_WEIGHTS_DIR'] + '/layer.pth')

    def load_weights(self, map_location='cpu'):
        self.load_state_dict(torch.load(dotenv_values()['ALIGN_WEIGHTS_DIR'] + '/layer.pth', map_location=map_location))


class AlignmentAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_encoder = FeatureEncoder()
        self.alignment_layer = AlignmentLayer()

    def forward(self, x):
        x = self.feature_encoder(x)
        return self.alignment_layer(x)

    def save_weights(self):
        torch.save(self.state_dict(), dotenv_values()['ALIGN_WEIGHTS_DIR'] + '/adapter.pth')

    def load_default(self):
        self.alignment_layer.load_default()
        self.feature_encoder.load_weights(dotenv_values()['FCN_WEIGHTS_DIR'])

    def load_weights(self, map_location='cpu'):
        if not os.path.exists(dotenv_values()['ALIGN_WEIGHTS_DIR'] + '/adapter.pth'):
            self.load_default()
            self.save_weights()
        else:
            self.load_state_dict(torch.load(dotenv_values()['ALIGN_WEIGHTS_DIR'] + '/adapter.pth', map_location=map_location))


class ModifiedEmbedding(nn.Module):
    def __init__(self, embedding):
        super().__init__()
        self.embedding = embedding
        self.adapter = AlignmentAdapter()
        self.adapter.load_weights()
        self.adapter.to(embedding.weight.device)
        self.signal_converter = IdConverter()

    def forward(self, x):
        if x.max() >= signal_token_id:
            text_part = x[x < signal_token_id].detach()
            signal_part = x[x >= signal_token_id].detach()
            text_output = self.embedding(text_part)
            signal_output = self.signal_converter.get_signal(signal_part, self.training)
            signal_output = self.adapter(signal_output).reshape(-1, llm_hidden_size)
            output = torch.zeros(x.size(0), x.size(1), llm_hidden_size, dtype=torch.bfloat16).to(x.device)
            output[x < signal_token_id] = text_output
            output[x >= signal_token_id] = signal_output
            return output
        else:
            return self.embedding(x)


def get_bearllm():
    hp = HyperParameters()
    config = AutoConfig.from_pretrained(f'{dotenv_values()['QWEN_WEIGHTS_DIR']}/config.json')
    model = AutoModelForCausalLM.from_pretrained(
        dotenv_values()['QWEN_WEIGHTS_DIR'],
        device_map=hp.device,
        torch_dtype="auto",
        trust_remote_code=True,
        config=config
    )
    embedding = model.get_input_embeddings()
    mod_embedding = ModifiedEmbedding(embedding)
    model.set_input_embeddings(mod_embedding)
    return model


def mod_xt_for_qwen(xt):
    text_part1 = '<|im_start|>system\n' + sys_prompt + '\n<|im_end|><|im_start|>user\n' + xt.split('#placeholder#')[0]
    text_part2 = xt.split('#placeholder#')[1] + '<|im_end|>\n<|im_start|>assistant\n'
    return text_part1, text_part2


def collate_fn(batch):
    input_ids = torch.nn.utils.rnn.pad_sequence([x['input_ids'] for x in batch], batch_first=True,
                                                padding_value=0)
    attention_mask = torch.nn.utils.rnn.pad_sequence([x['attention_mask'] for x in batch], batch_first=True,
                                                     padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence([x['labels'] for x in batch], batch_first=True,
                                             padding_value=-100)
    return {
        'input_ids': input_ids.detach(),
        'attention_mask': attention_mask.detach(),
        'labels': labels.detach()
    }


class FineTuningDataset(Dataset):
    def __init__(self):
        self.dataset = MBHMDataset()
        self.hp = HyperParameters()
        self.tokenizer = AutoTokenizer.from_pretrained(dotenv_values()['QWEN_WEIGHTS_DIR'])
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def __len__(self):
        return self.dataset.length

    def __getitem__(self, idx):
        xv, label, cid, xt, gt = self.dataset.__getitem__(idx)
        signal_ids = signal_token_id + torch.ones(description_len) * idx
        user_part1, user_part2 = mod_xt_for_qwen(xt)
        user_part1_ids = self.tokenizer(user_part1, return_tensors='pt', add_special_tokens=False).input_ids[0]
        user_part2_ids = self.tokenizer(user_part2, return_tensors='pt', add_special_tokens=False).input_ids[0]
        user_ids = torch.cat([user_part1_ids, signal_ids, user_part2_ids])
        gt_ids = self.tokenizer(gt, return_tensors='pt', add_special_tokens=False).input_ids[0]
        input_ids = torch.cat([user_ids, gt_ids, torch.ones(1) * self.tokenizer.pad_token_id])
        attention_mask = torch.ones_like(input_ids)
        labels = torch.cat([torch.ones_like(user_ids) * -100, gt_ids, torch.ones(1) * self.tokenizer.pad_token_id])
        return {
            'input_ids': input_ids.long().detach(),
            'attention_mask': attention_mask.long().detach(),
            'labels': labels.long().detach()
        }


def fine_tuning():
    hp = HyperParameters()
    model = get_bearllm()

    lora_config = LoraConfig(target_modules="all-linear",
                             task_type=TaskType.CAUSAL_LM,  # 任务类型
                             r=hp.r,
                             lora_alpha=hp.lora_alpha,
                             lora_dropout=hp.lora_dropout,
                             )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.train()

    dataset = FineTuningDataset()

    train_args = TrainingArguments(
        output_dir=dotenv_values()['LORA_WEIGHTS_DIR'],
        per_device_train_batch_size=hp.per_device_train_batch_size,
        gradient_accumulation_steps=hp.gradient_accumulation_steps,
        logging_steps=hp.logging_steps,
        num_train_epochs=hp.num_train_epochs,
        save_steps=hp.save_steps,
        learning_rate=hp.learning_rate,
        lr_scheduler_type=hp.lr_scheduler_type,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=dataset,
        eval_dataset=None,
        data_collator=collate_fn
    )

    trainer.train()
    trainer.model.save_pretrained(dotenv_values()['LORA_WEIGHTS_DIR'])


if __name__ == "__main__":
    fine_tuning()
