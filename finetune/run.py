import subprocess
import os

# set environment variable
DATA_PATH = "data/EvolInstruct-Code-80k.json"
OUTPUT_PATH = "output"
MODEL_PATH = "models/deepseek-coder-1.3b-instruct"

command = [
    "python", "finetune_deepseekcoder.py",
    "--model_name_or_path", MODEL_PATH,
    "--data_path", DATA_PATH,
    "--output_dir", OUTPUT_PATH,
    "--num_train_epochs", "3",
    "--model_max_length", "512",
    "--per_device_train_batch_size", "4",
    "--per_device_eval_batch_size", "4",
    "--gradient_accumulation_steps", "1",
    "--evaluation_strategy", "no",
    "--save_strategy", "steps",
    "--save_steps", "500",
    "--save_total_limit", "1",
    "--learning_rate", "3e-5",
    "--warmup_steps", "100",
    "--logging_steps", "1",
    "--lr_scheduler_type", "cosine",
    "--gradient_checkpointing",
    "--report_to", "tensorboard",
    "--max_grad_norm", "1.0",
    "--fp16"
]


subprocess.run(command, check=True)
