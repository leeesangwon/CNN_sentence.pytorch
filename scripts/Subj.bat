cd ../src
python main.py --dataset=Subj --dataset_folder="../resource/data/Subj" --model=random
python main.py --dataset=Subj --dataset_folder="../resource/data/Subj" --model=static
python main.py --dataset=Subj --dataset_folder="../resource/data/Subj" --model=non-static
python main.py --dataset=Subj --dataset_folder="../resource/data/Subj" --model=multi-channel
cd ../scripts