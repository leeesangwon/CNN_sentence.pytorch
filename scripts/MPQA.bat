cd ../src
python main.py --dataset=MPQA --dataset_folder="../resource/data/MPQA" --model=random
python main.py --dataset=MPQA --dataset_folder="../resource/data/MPQA" --model=static
python main.py --dataset=MPQA --dataset_folder="../resource/data/MPQA" --model=non-static
python main.py --dataset=MPQA --dataset_folder="../resource/data/MPQA" --model=multi-channel
cd ../scripts