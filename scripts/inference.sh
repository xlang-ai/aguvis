MODEL_PATH="Please provide the path to the model"
IMAGE_PATH="src/aguvis/serve/examples/AndroidControl.png"
INSTRUCTION="In the BBC News app , Turn ON the news alert notification for the BBC News app."

python src/aguvis/serve/cli.py \
    --model_path "$MODEL_PATH" \
    --image_path "$IMAGE_PATH" \
    --instruction "$INSTRUCTION" \
    --temperature 0 \
    --max_new_tokens 1024 \
    --mode self-plan \
    --device cuda \
    # --previous_actions "$PREVIOUS_ACTIONS" \
    # --low_level_instruction "$LOW_LEVEL_INSTRUCTION" \
