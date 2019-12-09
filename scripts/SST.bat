cd ../src
python main.py --dataset=SST1 --dataset_folder="../resource/data/SST1" --model=random
python main.py --dataset=SST1 --dataset_folder="../resource/data/SST1" --model=static
python main.py --dataset=SST1 --dataset_folder="../resource/data/SST1" --model=non-static
python main.py --dataset=SST1 --dataset_folder="../resource/data/SST1" --model=multi-channel
python main.py --dataset=SST2 --dataset_folder="../resource/data/SST2" --model=random
python main.py --dataset=SST2 --dataset_folder="../resource/data/SST2" --model=static
python main.py --dataset=SST2 --dataset_folder="../resource/data/SST2" --model=non-static
python main.py --dataset=SST2 --dataset_folder="../resource/data/SST2" --model=multi-channel
cd ../scripts
