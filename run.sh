#!/bin/bash

MODEL_VERSION="$1"

SEED=42

# Step 1: Run setup, including:
# * Download Dataset
# * Split into Train, Validation and Test subsets
# * Calculate mean and var for feature normalization with training set
echo "Running setup.py..."
python setup.py --model_version "$MODEL_VERSION" --seed "$SEED"

# Step 2: Nested loop to run training with various parameters
# and get validation accuracy
n_neurons_list=(4 8 16 32)
activation_list=("sigmoid" "relu")
optimizer_list=("sgd" "rmsprop" "adam")

echo "Starting training runs..."
for n_neurons in "${n_neurons_list[@]}"; do
  for activation in "${activation_list[@]}"; do
    for optimizer in "${optimizer_list[@]}"; do
      for i in {1..3}; do
        echo "Training with n_neurons=$n_neurons, activation=$activation, optimizer=$optimizer, run=$i"
        python train.py --n_neurons "$n_neurons" --activation "$activation" --optimizer "$optimizer" --model_version "$MODEL_VERSION" 2>/dev/null
      done
    done
  done
done

# Step 3: Run evaluation, including:
# * find model with highest validation accuracy
# * measure the test accuracy
echo "Running evaluation..."
python evaluate.py --model_version "$MODEL_VERSION" 2>/dev/null
