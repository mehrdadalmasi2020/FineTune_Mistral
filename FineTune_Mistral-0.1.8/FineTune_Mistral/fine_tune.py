import gc
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments
)
from peft import LoraConfig, PeftModel
from random import randrange
from trl import SFTTrainer  # Correct import for the SFTTrainer

class FineTune_Mistral:
    def __init__(self, dataset_path, cache_dir):
        self.dataset_path = dataset_path
        self.cache_dir = cache_dir
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", cache_dir=self.cache_dir, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Set padding token to eos token
        self.tokenizer.padding_side = 'right'  # Ensure padding is on the right
        self.trained_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", cache_dir=self.cache_dir).to("cuda")
        self.dataset = load_dataset('json', data_files=self.dataset_path, split='train')

    def train_model(self, output_dir, num_train_epochs=3, per_device_train_batch_size=2, per_device_eval_batch_size=1, max_seq_length=None):
        lora_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "lm_head",
            ],
            bias="none",
            task_type="CAUSAL_LM",
        )

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            logging_dir=f"{output_dir}/logs",
            logging_steps=100,
            save_steps=1000,
            warmup_ratio=0.03,
            report_to="tensorboard"
        )

        trainer = SFTTrainer(
            model=self.trained_model,
            train_dataset=self.dataset,
            peft_config=lora_config,
            dataset_text_field="text",  # Assuming 'text' is the field name containing the text data
            max_seq_length=max_seq_length,  # Pass None or specify a maximum sequence length
            tokenizer=self.tokenizer,
            args=training_args
        )

        trainer.train()
        trainer.model.save_pretrained(output_dir)
        self.trained_model = trainer.model

    def generate_response(self, question, max_new_tokens=500, temperature=0.1):
        prompt = f"""You will be provided with a question. You must provide only a single answer. You must not provide additional questions and answers.
        Question:
        {question}
        """
        model_input = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            generated_code = self.trained_model.generate(**model_input, max_new_tokens=max_new_tokens, pad_token_id=0, temperature=temperature)
            generated_code = self.tokenizer.decode(generated_code[0], skip_special_tokens=True)
        response = generated_code.split("Question:")[1]
        if "Answer: " in response:
            return response.split("Answer:")[1].replace("\n", " ").strip()
        else:
            if len(str(response.split("\n")[3])) < 5:
                return response
            return response.split("\n")[3]

    def clean_up(self):
        del self.tokenizer
        del self.trained_model
        gc.collect()
        torch.cuda.empty_cache()

    def get_random_sample(self):
        return self.dataset[randrange(len(self.dataset))]
