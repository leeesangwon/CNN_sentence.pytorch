cd ../src
python main.py --dataset=MR --dataset_folder="../resource/data/MR" --model=random
python main.py --dataset=MR --dataset_folder="../resource/data/MR" --model=static
python main.py --dataset=MR --dataset_folder="../resource/data/MR" --model=non-static
python main.py --dataset=MR --dataset_folder="../resource/data/MR" --model=multi-channel
cd ../scripts