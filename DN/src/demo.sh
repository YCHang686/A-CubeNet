
#Train_noise level_10
#python main.py --template ACubeNet --save ACubeNet_DN10 --reset --quality 1
#Train_noise level_30
#python main.py --template ACubeNet --save ACubeNet_DN30 --reset --quality 3
#Train_noise level_50
#python main.py --template ACubeNet --save ACubeNet_DN50 --reset --quality 5
#Train_noise level_70
#python main.py --template ACubeNet --save ACubeNet_DN70 --reset --quality 7

#Test_noise level_10
#python main.py --template ACubeNet --data_test BSD68+Kodak24 --pre_train  ../experiment/ACubeNet_DN10/model/model_best.pt --test_only --save_results --save_gt --quality 1
#Test_noise level_30
#python main.py --template ACubeNet --data_test BSD68+Kodak24 --pre_train  ../experiment/ACubeNet_DN30/model/model_best.pt --test_only --save_results --save_gt --quality 3
#Test_noise level_50
#python main.py --template ACubeNet --data_test BSD68+Kodak24 --pre_train  ../experiment/ACubeNet_DN50/model/model_best.pt --test_only --save_results --save_gt --quality 5
#Test_noise level_70
#python main.py --template ACubeNet --data_test BSD68+Kodak24 --pre_train  ../experiment/ACubeNet_DN70/model/model_best.pt --test_only --save_results --save_gt --quality 7

