python mcts_cot.py --model_name meta-llama/Llama-2-13b-hf \
    --use_peft --batch_size 4 --gradient_accumulation_steps 2 --load_in_8bit --num_train_epochs 10 --learning_rate 2.0e-5