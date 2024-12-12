from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
import pandas as pd
from tqdm import tqdm


def data_load():
    """Load and return a subset of the dataset."""
    data = pd.read_json("data/EvolInstruct-Code-80k.json")
    return data.iloc[10000:10010].reset_index(drop=True)


def build_instruction_prompt(instruction: str):
    """
    Build a formatted prompt for the model to generate responses based on instructions.

    Args:
        instruction (str): The instruction string.

    Returns:
        str: Formatted prompt.
    """
    return '''
You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.
### Instruction:
{0}
### Response:
'''.format(instruction.strip()).lstrip()


def generate_responses(model, tokenizer, instructions, device):
    """
    Generate responses for a list of instructions using the provided model and tokenizer.

    Args:
        model (AutoModelForCausalLM): The model used for generation.
        tokenizer (AutoTokenizer): The tokenizer to preprocess inputs.
        instructions (list of str): A list of instructions to generate responses for.
        device (torch.device): The device on which the model should run.

    Returns:
        list of str: A list of generated responses.
    """
    responses = []

    # Ensure pad_token_id and eos_token_id are set correctly in the model's configuration
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if model.config.eos_token_id is None:
        model.config.eos_token_id = tokenizer.eos_token_id

    generation_config = GenerationConfig(
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=100,  # Use max_new_tokens instead of max_length
        do_sample=False,
        num_beams=1
    )

    for instruction in tqdm(instructions, desc="Generating Responses"):
        # Create a formatted prompt for this instruction
        prompt = build_instruction_prompt(instruction)

        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        outputs = model.generate(**inputs, generation_config=generation_config)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(response)
        responses.append(response)

    return responses


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(
        "models/deepseek-coder-1.3b-instruct",
        trust_remote_code=True,
        pad_token="<｜end▁of▁sentence｜>",
        bos_token="<｜begin▁of▁sentence｜>",
        eos_token="<|EOT|>"
    )

    model = AutoModelForCausalLM.from_pretrained("output/checkpoint-7500", trust_remote_code=True).cuda()

    print(f"Pad Token ID: {model.config.pad_token_id}")
    print(f"EOS Token ID: {model.config.eos_token_id}")

    data = data_load()
    instruction_list = data["instruction"]
    turn_output_list = data["output"]

    # Generate responses with prompts
    predicted_outputs = generate_responses(model, tokenizer, instruction_list, device=model.device)

    results_df = pd.DataFrame({
        'Instruction': instruction_list,
        'Real Output': turn_output_list,
        'Predicted Output': predicted_outputs
    })

    results_df.to_csv('model_predictions.csv', index=False)

    print("Results have been saved to 'model_predictions.csv'.")