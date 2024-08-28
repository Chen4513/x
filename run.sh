#!/bin/bash

# Path to the YAML file

SRC=$(pwd)

# Check if the YAML file exists

MODES=("llm" "baseline")

BASE_COMMAND="python hopper_tutorial.py"

TEACHER_MODEL_PATHS=("LLM/RM/models/reward_model-mixtral-06err.pth" "LLM/RM/models/reward_model-mixtralx5-06err.pth")
for i in {1..3}; do
    for mode in "${MODES[@]}"; do

        if [[ "$mode" == *"baseline"* ]]; then
            mode_name="ds"

        elif [[ "$mode" == *"llm"* ]]; then
            mode_name="sd"
        fi

        for TEACHER_MODEL_PATH in "${TEACHER_MODEL_PATHS[@]}"; do
            if [[ "$TEACHER_MODEL_PATH" == *"mixtral-05err.pth"* ]]; then
                name="1query"
                err="05"
                
                mean="0.4477"
                std="0.3754"

            elif [[ "$TEACHER_MODEL_PATH" == *"mixtralx5-05err"* ]]; then
                name="5query"
                err="05"
                
                mean="0.3663"
                std="0.1583"

            elif [[ "$TEACHER_MODEL_PATH" == *"mixtral-06err.pth"* ]]; then
                name="1query"
                err="06"
                
                mean="0.8346"
                std="0.3241"

            elif [[ "$TEACHER_MODEL_PATH" == *"mixtralx5-06err"* ]]; then
                name="5query"
                err="06"
                
                mean="1.2657"
                std="0.4957"

            elif [[ "$TEACHER_MODEL_PATH" == *"mixtral-07err.pth"* ]]; then
                name="1query"
                err="07"

                mean="0.8805"
                std="0.5373"                

            elif [[ "$TEACHER_MODEL_PATH" == *"mixtralx5-07err"* ]]; then
                name="5query"
                err="07"

                mean="1.9372"
                std="0.8664"                  
            
            elif [[ "$TEACHER_MODEL_PATH" == *"mixtral-08err.pth"* ]]; then
                name="1query"
                err="08"

                mean="2.9935"
                std="2.1950"                

            elif [[ "$TEACHER_MODEL_PATH" == *"mixtralx5-08err"* ]]; then
                name="5query"
                err="08"

                mean="3.6848"
                std="2.0983"    
            fi
            echo $BASE_COMMAND --name="${err}err-${mode_name}-${name}_trial$i" --mode="${mode}" --teacher_model_path="${TEACHER_MODEL_PATH}" --score_mean=${mean} --score_std=${std}
            $BASE_COMMAND --name="${err}err-${mode_name}-${name}_trial$i" --mode="${mode}" --teacher_model_path="${TEACHER_MODEL_PATH}" --score_mean=${mean} --score_std=${std}
        done
    done
done