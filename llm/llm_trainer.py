from unsloth import FastLanguageModel
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import get_scheduler
from peft import PeftModel
import numpy as np
import pickle
import os

class ConflictResponseDataset(Dataset):
    def __init__(self, data, tokenizer, system_prompt, max_length=1024):
        self.data = data
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prompt, response, reward = self.data[idx]

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        tokenized = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "reward": torch.tensor(reward, dtype=torch.float32)
        }

class LLMTrainer:
    def __init__(self, model_name, device="cuda", use_lora=True, lora_r=8, lora_alpha=16, lora_dropout=0.1):
        self.device = torch.device(device)

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=1024,
            dtype=None,
            load_in_4bit=True
        )

        if use_lora:
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=['o_proj', 'down_proj', 'up_proj', 'k_proj', 'gate_proj', 'v_proj', 'q_proj']
            )

        self.model.print_trainable_parameters()
        self.logs = []

    def prepare_dataset(self, data, system_prompt, max_length):
        return ConflictResponseDataset(data, self.tokenizer, system_prompt=system_prompt, max_length=max_length)

    def train(self, dataset, epochs=3, batch_size=4, lr=5e-5, log_curiosity=False,
              model_save_path="checkpoints/llm", log_path="checkpoints/llm_training_log.pkl"):

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=len(dataloader) * epochs
        )

        self.model.train()
        for epoch in range(epochs):
            if epoch + 1 == 10:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 2e-5
                    print(f"Epoch {epoch+1}: Learning rate adjusted to 2e-5")

            total_loss = 0
            curiosity_log = []

            num_batches = len(dataloader)
            progress_checkpoints = {int(num_batches * p / 100) for p in range(5, 101, 5)}

            for step, batch in enumerate(dataloader):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["input_ids"]
                )

                base_loss = outputs.loss
                ###
                ###batch["reward"] = torch.full_like(batch["reward"], 0.7)
                ###
                weighted_loss = base_loss * batch["reward"].mean()

                weighted_loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                total_loss += weighted_loss.item()
                if log_curiosity:
                    curiosity_log.append(batch["reward"].mean().item())

                if step in progress_checkpoints:
                    percent_done = int((step / num_batches) * 100)
                    print(f"Epoch {epoch+1} progress: {percent_done}% complete")

            avg_loss = total_loss / len(dataloader)
            avg_c = np.mean(curiosity_log) if log_curiosity else None
            std_c = np.std(curiosity_log) if log_curiosity else None

            log_entry = {
                "epoch": epoch + 1,
                "loss": avg_loss,
                "avg_curiosity": avg_c,
                "std_curiosity": std_c
            }
            self.logs.append(log_entry)

            print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")
            if log_curiosity:
                print(f"          Avg Curiosity = {avg_c:.6f}, Std = {std_c:.6f}")

            os.makedirs(model_save_path, exist_ok=True)
            self.save(model_save_path)

            if log_path:
                with open(os.path.join(os.path.dirname(__file__), log_path), "wb") as pf:
                    pickle.dump(self.logs, pf)

    def save(self, path):
        self.model.save_pretrained(os.path.join(os.path.dirname(__file__), path))
        self.tokenizer.save_pretrained(os.path.join(os.path.dirname(__file__), path))

    def load(self, path):
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=path,
            max_seq_length=1024,
            dtype=None,
            load_in_4bit=True
        )
