#!/bin/bash

# Path to the YAML file

SRC=$(pwd)

BASE_COMMAND="python hopper_tutorial.py"

# TEACHER_MODEL_PATHS=("$SRC/LLM/RM/models/reward_model_llama3.pth" "$SRC/LLM/RM/models/reward_model6knew.pth" "$SRC/LLM/RM/models/reward_model_gpt3.pth" "$SRC/LLM/RM/models/reward_model_mixtral6000.pth")
# TEACHER_MODEL_PATHS=("$SRC/LLM/RM/models/reward_model_llama3.pth" "$SRC/LLM/RM/models/reward_model6knew.pth" "$SRC/LLM/RM/models/reward_model_gpt3.pth")
# TEACHER_MODEL_PATHS=("$SRC/LLM/RM/models/reward_model_llama3.pth")
# TEACHER_MODEL_PATHS=("reward_model_llama1k.pth" "reward_model_gpt1k.pth" "reward_model_mixtral1k.pth" "reward_model_llama8b1k.pth" "reward_model_mixtral1kx5.pth")

# # MODES=("llm" "baseline")
# MODES=("llm" "baseline")

# for i in {1..5}; do
#     for TEACHER_MODEL_PATH in "${TEACHER_MODEL_PATHS[@]}"; do
#         for mode in "${MODES[@]}"; do
#             if [[ "$mode" == "llm" ]]; then
#                 score_mean=0
#                 score_std=1
#             fi
#             if [[ "$TEACHER_MODEL_PATH" == *"mixtral1kx5"* ]]; then
#                 experiment_name="mixtralx5"
#                 if [[ "$mode" == "baseline" ]]; then
#                     score_mean=8.3397
#                     score_std=4.8809
#                 fi
#             elif [[ "$TEACHER_MODEL_PATH" == *"mixtral"* ]]; then
#                 experiment_name="mixtralx1"
#                 if [[ "$mode" == "baseline" ]]; then
#                     score_mean=7.4673
#                     score_std=4.6807
#                 fi
#             elif [[ "$TEACHER_MODEL_PATH" == *"llama8b"* ]]; then
#                 experiment_name="llama8b"
#                 if [[ "$mode" == "baseline" ]]; then
#                     score_mean=1.8340
#                     score_std=1.7684
#                 fi
#             elif [[ "$TEACHER_MODEL_PATH" == *"llama1k"* ]]; then
#                 experiment_name="llama"
#                 if [[ "$mode" == "baseline" ]]; then
#                     score_mean=17.8495
#                     score_std=14.4532
#                 fi
#             elif [[ "$TEACHER_MODEL_PATH" == *"gpt"* ]]; then
#                 experiment_name="gpt"
#                 if [[ "$mode" == "baseline" ]]; then
#                     score_mean=4.4382
#                     score_std=4.5797
#                 fi
#             fi
#             $BASE_COMMAND --name="hopper_${experiment_name}_${mode}_trial$i" --mode=$mode --teacher_model_path=$TEACHER_MODEL_PATH --score_mean=$score_mean --score_std=$score_std
#         done
#     done
# done

for i in {1..3}; do
    $BASE_COMMAND --name="hopper_default_trial$i" --mode="default"
done

# yq e '.BASE_CONFIG.reward_clipping_func = "tanh"' -i "$FILE"

# # MAX_R=(1 5 10 15 20 25 30 35 40 45)
# MAX_R=(1 10 20 30 45)
# for i in "${MAX_R[@]}"
# do
#     # Update values
#     yq e ".BASE_CONFIG.max_reward_scale = $i" -i "$FILE"
#     echo "YAML file updated successfully."

#     for j in {1..3}
#     do
#         # Run the command with the current iteration number as the logging_name
#         $BASE_COMMAND --logging_name="env0-gt_tanh_scale$i-$j"
#         echo "Completed run $i"
#     done
# done

# yq e '.BASE_CONFIG.reward_clipping_func = "sigmoid"' -i "$FILE"

# # MAX_R=(1 5 10 15 20 25 30 35 40 45)
# MAX_R=(1 10 20 30 45)
# for i in "${MAX_R[@]}"
# do
#     # Update values
#     yq e ".BASE_CONFIG.max_reward_scale = $i" -i "$FILE"
#     echo "YAML file updated successfully."

#     for j in {1..3}
#     do
#         # Run the command with the current iteration number as the logging_name
#         $BASE_COMMAND --logging_name="env0-gt_sig_scale$i-$j"
#         echo "Completed run $i"
#     done
# done