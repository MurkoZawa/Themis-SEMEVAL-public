::increasing lora r
python train.py --name_llm "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --name_img_embed "openai/clip-vit-large-patch14" --batch_size 2 --use_lora True --lora_alpha 8 --lora_r 8 --lora_dropout 0.2 --merge_tokens 0
python train.py --name_llm "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --name_img_embed "openai/clip-vit-large-patch14" --batch_size 2 --use_lora True --lora_alpha 8 --lora_r 16 --lora_dropout 0.2 --merge_tokens 0
python train.py --name_llm "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --name_img_embed "openai/clip-vit-large-patch14" --batch_size 2 --use_lora True --lora_alpha 8 --lora_r 32 --lora_dropout 0.2 --merge_tokens 0
::increasing lora dropout
python train.py --name_llm "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --name_img_embed "openai/clip-vit-large-patch14" --batch_size 2 --use_lora True --lora_alpha 8 --lora_r 8 --lora_dropout 0.3 --merge_tokens 0
python train.py --name_llm "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --name_img_embed "openai/clip-vit-large-patch14" --batch_size 2 --use_lora True --lora_alpha 8 --lora_r 8 --lora_dropout 0.4 --merge_tokens 0
python train.py --name_llm "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --name_img_embed "openai/clip-vit-large-patch14" --batch_size 2 --use_lora True --lora_alpha 8 --lora_r 8 --lora_dropout 0.5 --merge_tokens 0
python train.py --name_llm "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --name_img_embed "openai/clip-vit-large-patch14" --batch_size 2 --use_lora True --lora_alpha 8 --lora_r 8 --lora_dropout 0.6 --merge_tokens 0
::increasing number of tokens
python train.py --name_llm "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --name_img_embed "openai/clip-vit-large-patch14" --batch_size 2 --use_lora True --lora_alpha 8 --lora_r 8 --lora_dropout 0.2 --merge_tokens 64
python train.py --name_llm "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --name_img_embed "openai/clip-vit-large-patch14" --batch_size 2 --use_lora True --lora_alpha 8 --lora_r 8 --lora_dropout 0.2 --merge_tokens 96
python train.py --name_llm "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --name_img_embed "openai/clip-vit-large-patch14" --batch_size 2 --use_lora True --lora_alpha 8 --lora_r 8 --lora_dropout 0.2 --merge_tokens 128
python train.py --name_llm "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --name_img_embed "openai/clip-vit-large-patch14" --batch_size 2 --use_lora True --lora_alpha 8 --lora_r 8 --lora_dropout 0.2 --merge_tokens 192
::not using lora
python train.py --name_llm "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --name_img_embed "openai/clip-vit-large-patch14" --batch_size 2 --use_lora False --merge_tokens 0 --lora_alpha 8 --lora_r 8 --lora_dropout 0.2 
::not using lora and increasing number of tokens
python train.py --name_llm "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --name_img_embed "openai/clip-vit-large-patch14" --batch_size 2 --use_lora False --merge_tokens 64 --lora_alpha 8 --lora_r 8 --lora_dropout 0.2 
python train.py --name_llm "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --name_img_embed "openai/clip-vit-large-patch14" --batch_size 2 --use_lora False --merge_tokens 96 --lora_alpha 8 --lora_r 8 --lora_dropout 0.2 
python train.py --name_llm "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --name_img_embed "openai/clip-vit-large-patch14" --batch_size 2 --use_lora False --merge_tokens 128 --lora_alpha 8 --lora_r 8 --lora_dropout 0.2 
python train.py --name_llm "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --name_img_embed "openai/clip-vit-large-patch14" --batch_size 2 --use_lora False --merge_tokens 192 --lora_alpha 8 --lora_r 8 --lora_dropout 0.2 


python train.py --name_llm "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --name_img_embed "openai/clip-vit-large-patch14" --batch_size 2 --use_lora True --lora_alpha 8 --lora_r 8 --lora_dropout 0.3 --merge_tokens 0
python train.py --name_llm "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --name_img_embed "openai/clip-vit-large-patch14" --batch_size 2 --use_lora True --lora_alpha 8 --lora_r 8 --lora_dropout 0.4 --merge_tokens 0
python train.py --name_llm "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --name_img_embed "openai/clip-vit-large-patch14" --batch_size 2 --use_lora True --lora_alpha 8 --lora_r 8 --lora_dropout 0.5 --merge_tokens 0
python train.py --name_llm "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --name_img_embed "openai/clip-vit-large-patch14" --batch_size 2 --use_lora True --lora_alpha 8 --lora_r 8 --lora_dropout 0.6 --merge_tokens 0

python train.py --name_llm "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --name_img_embed "openai/clip-vit-large-patch14" --batch_size 2 --use_lora True --lora_alpha 8 --lora_r 8 --lora_dropout 0.4 --merge_tokens 0 --number_of_epochs 15
python train.py --name_llm "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --name_img_embed "openai/clip-vit-large-patch14" --batch_size 2 --use_lora True --lora_alpha 8 --lora_r 8 --lora_dropout 0.4 --merge_tokens 0 --number_of_epochs 20
python train.py --name_llm "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --name_img_embed "openai/clip-vit-large-patch14" --batch_size 2 --use_lora True --lora_alpha 8 --lora_r 8 --lora_dropout 0.4 --merge_tokens 0 --number_of_epochs 25


