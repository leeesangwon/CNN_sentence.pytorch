cd ../src
python main.py --dataset=MR --dataset_folder="../resource/MR" --model=random
python main.py --dataset=MR --dataset_folder="../resource/MR" --model=static
python main.py --dataset=MR --dataset_folder="../resource/MR" --model=non-static
python main.py --dataset=MR --dataset_folder="../resource/MR" --model=multi-channel
cd ../scripts
