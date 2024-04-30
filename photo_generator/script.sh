#!/bin/bash

# Set the Python script path
SCRIPT_PATH="generate.py"

# Set default values for command line arguments
PROMPT="Render a pair of stylish sunglasses resting on a tropical-themed background, with dappled sunlight filtering through palm leaves. Use a low camera angle and shallow depth of field to create a sense of warmth and relaxation."
NEGATIVE_PROMPT=""
SEED=723645
CUSTOM_WIDTH=1024
CUSTOM_HEIGHT=1024
GUIDANCE_SCALE=7.0
NUM_INFERENCE_STEPS=30
SAMPLER="DPM++ 2M SDE Karras"
ASPECT_RATIO_SELECTOR="1024 x 1024"
USE_UPSCALER=false
UPSCALER_STRENGTH=0.55
UPSCALE_BY=1.5

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --prompt=*|-p=*)
            PROMPT="${1#*=}"
            ;;
        --negative_prompt=*|-n=*)
            NEGATIVE_PROMPT="${1#*=}"
            ;;
        --seed=*|-s=*)
            SEED="${1#*=}"
            ;;
        --custom_width=*|-w=*)
            CUSTOM_WIDTH="${1#*=}"
            ;;
        --custom_height=*|-h=*)
            CUSTOM_HEIGHT="${1#*=}"
            ;;
        --guidance_scale=*|-g=*)
            GUIDANCE_SCALE="${1#*=}"
            ;;
        --num_inference_steps=*|-i=*)
            NUM_INFERENCE_STEPS="${1#*=}"
            ;;
        --sampler=*|-m=*)
            SAMPLER="${1#*=}"
            ;;
        --aspect_ratio_selector=*|-a=*)
            ASPECT_RATIO_SELECTOR="${1#*=}"
            ;;
        --use_upscaler|-u)
            USE_UPSCALER=true
            ;;
        --upscaler_strength=*|-r=*)
            UPSCALER_STRENGTH="${1#*=}"
            ;;
        --upscale_by=*|-b=*)
            UPSCALE_BY="${1#*=}"
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
    shift
done

# Construct the command to run the Python script
COMMAND="python $SCRIPT_PATH --prompt='$PROMPT'"
COMMAND+=" --negative_prompt='$NEGATIVE_PROMPT'"
COMMAND+=" --seed=$SEED"
COMMAND+=" --custom_width=$CUSTOM_WIDTH"
COMMAND+=" --custom_height=$CUSTOM_HEIGHT"
COMMAND+=" --guidance_scale=$GUIDANCE_SCALE"
COMMAND+=" --num_inference_steps=$NUM_INFERENCE_STEPS"
COMMAND+=" --sampler='$SAMPLER'"
COMMAND+=" --aspect_ratio_selector='$ASPECT_RATIO_SELECTOR'"
if $USE_UPSCALER; then
    COMMAND+=" --use_upscaler"
    COMMAND+=" --upscaler_strength=$UPSCALER_STRENGTH"
    COMMAND+=" --upscale_by=$UPSCALE_BY"
fi

# Execute the Python script
eval "$COMMAND"