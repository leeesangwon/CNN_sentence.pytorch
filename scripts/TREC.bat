cd ../src
python main.py --dataset=TREC --dataset_folder="../resource/data/TREC" --model=random
python main.py --dataset=TREC --dataset_folder="../resource/data/TREC" --model=static
python main.py --dataset=TREC --dataset_folder="../resource/data/TREC" --model=non-static
python main.py --dataset=TREC --dataset_folder="../resource/data/TREC" --model=multi-channel
cd ../scripts
