# for train enc
python experiments/fastst/main.py --device cuda:0 --dataset CA --years 2019 --seed 2024 --bs 1 --max_epochs 300 --model_name fastst --patience 100 --mode train_enc --learning_rate 0.001 --enc_description enc --n_increments 0 --enc_hidden_dim 32 --max_layer 2 --down_method meanpool

# for train pred
python experiments/fastst/main.py --device cuda:0 --dataset CA --years 2019 --seed 2024 --bs 32 --max_epochs 100 --model_name fastst --patience 50 --mode train_pred --pred_description gwnet --if_print True --base_model GWNET --max_layer 2

# for train agent
python experiments/fastst/main.py --device cuda:0 --dataset CA --years 2019 --seed 2024 --if_print True --bs 1 --max_epochs 151 --model_name fastst --patience 151 --mode train_agent --enc_seed 2024 --enc_description enc --pred_seed 2024 --pred_description gwnet --agent_description agent --tradeoff 5 --max_layer 2 --enc_hidden_dim 32 --agent_hidden_dim 64 --down_method meanpool --n_increments 0 --wdecay 0.00001 --learn_times 1 --learn_cnts 8 --buffer_size 1200 --learning_rate 0.0001 --skip_time_cost 0.01 --base_model GWNET