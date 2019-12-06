cd ../src
python main.py --dataset=CR --dataset_folder="../resource/data/CR" --model=random
python main.py --dataset=CR --dataset_folder="../resource/data/CR" --model=static
python main.py --dataset=CR --dataset_folder="../resource/data/CR" --model=non-static
python main.py --dataset=CR --dataset_folder="../resource/data/CR" --model=multi-channel
cd ../scripts