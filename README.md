# DeepWalkSG
This is a python code for DeepWalk with SkipGram using PyTorch with the following requirements:
torch                   1.5.0               
torchvision             0.6.0 

To run the code, use the following script:
python main.py --dataset DBLP --number_walks 80 --window_size 5 --negative_samples 5 --walk_length 40 --dimension 128 --iterations 5 --workers 10 --batch_size 10000 --train 1

This is an implmentation for DeepWalk, don't forget to add the following reference.
Perozzi, Bryan, Rami Al-Rfou, and Steven Skiena. "Deepwalk: Online learning of social representations." Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining. 2014.
