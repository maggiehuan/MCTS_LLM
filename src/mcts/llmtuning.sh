python llmtuning.py --model_name meta-llama/Llama-2-13b-hf \
    --dataset_name timdettmers/openassistant-guanaco \
    --use_peft --batch_size 4 --gradient_accumulation_steps 2 --load_in_8bit --num_train_epochs 50 --learning_rate 5.0e-5