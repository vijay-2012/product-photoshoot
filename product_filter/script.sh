#!/bin/bash

# Set the Python script path
SCRIPT_PATH="apply_filter.py"

# Parse the provided arguments
INPUT_DIR="products_unfiltered"
TARGET_DIR="products_filtered"
NON_RELEVANT_DIR="non_revelent_products"
LABELS="Shoe, Sneaker, Bottle, Cup, Sandal, Perfume, Toy, Sunglasses, Car, Water Bottle, Chair, Office Chair, Can, Cap, Hat, Couch, Wristwatch, Glass, Bag, Handbag, Baggage, Suitcase, Headphones, Jar, Vase"
THRESHOLD=0.3

# Construct the command to run the Python script
COMMAND="python $SCRIPT_PATH --input_dir='$INPUT_DIR' --target_dir='$TARGET_DIR' --non_relevant_dir='$NON_RELEVANT_DIR' --labels $LABELS --threshold=$THRESHOLD"

# Execute the Python script
eval "$COMMAND"