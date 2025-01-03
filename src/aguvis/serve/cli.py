import argparse
from io import BytesIO
from typing import List, Literal, Optional

import requests
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor

from aguvis.constants import agent_system_message, chat_template, grounding_system_message, until, user_instruction


def load_image(image_file):
    if image_file.startswith("http://") or image_file.startswith("https://"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_pretrained_model(model_path):
    model = Qwen2VLForConditionalGeneration.from_pretrained(model_path)
    processor = Qwen2VLProcessor.from_pretrained(model_path)
    tokenizer = processor.tokenizer
    return model, processor, tokenizer


def generate_response(
    model,
    processor,
    tokenizer,
    image: Image.Image,
    instruction: str,
    previous_actions: Optional[str | List[str]] = None,
    low_level_instruction: Optional[str] = None,
    mode: Literal["self-plan", "force-plan", "grounding"] = "self-plan",
    temperature: float = 0,
    max_new_tokens: int = 1024,
):
    system_message = {
        "role": "system",
        "content": grounding_system_message if mode == "grounding" else agent_system_message,
    }

    if isinstance(previous_actions, list):
        # Convert previous actions to string. Expecting the format:
        # ["Step 1: Swipe up", "Step 2: Click on the search bar"]
        previous_actions = "\n".join(previous_actions)
    if not previous_actions:
        previous_actions = "None"
    user_message = {
        "role": "user",
        "content": user_instruction.format(
            overall_goal=instruction,
            previous_actions=previous_actions,
            low_level_instruction=low_level_instruction,
        ),
    }

    if low_level_instruction:
        # If low-level instruction is provided
        # We enforce using "Action: {low_level_instruction} to guide generation"
        recipient_text = f"<|im_start|>assistant<|recipient|>all\nAction: {low_level_instruction}\n"
    elif mode == "force-plan":
        recipient_text = "<|im_start|>assistant<|recipient|>all\nThought: "
    elif mode == "grounding":
        recipient_text = "<|im_start|>assistant<|recipient|>all\n"
    elif mode == "self-plan":
        recipient_text = ""
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # Generate response
    messages = [system_message, user_message]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, chat_template=chat_template
    )
    text += recipient_text
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)
    cont = model.generate(**inputs, temperature=temperature, max_new_tokens=max_new_tokens)

    cont_toks = cont.tolist()[0][len(inputs.input_ids[0]) :]
    text_outputs = tokenizer.decode(cont_toks, skip_special_tokens=True).strip()
    for term in until:
        if len(term) > 0:
            text_outputs = text_outputs.split(term)[0]
    return text_outputs


def main(args):
    model, processor, tokenizer = load_pretrained_model(args.model_path)
    model.to(args.device)
    model.tie_weights()
    image = load_image(args.image_path)
    response = generate_response(
        model,
        processor,
        tokenizer,
        image,
        args.instruction,
        args.previous_actions,
        args.low_level_instruction,
        args.mode,
        args.temperature,
        args.max_new_tokens,
    )
    print(response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image_path", type=str, default="examples/AndroidControl.png")
    parser.add_argument(
        "--instruction",
        type=str,
        default="In the BBC News app , Turn ON the news alert notification for the BBC News app.",
    )
    parser.add_argument("--previous_actions", type=str, required=False)
    parser.add_argument("--low_level_instruction", type=str, required=False)
    parser.add_argument("--mode", type=str, default="self-plan")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    args = parser.parse_args()
    main(args)
