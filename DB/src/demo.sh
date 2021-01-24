
#Train_quality_10
#python main.py --template ACubeNet --save ACubeNet_DB10 --reset --quality 1
#Train_quality_20
#python main.py --template ACubeNet --save ACubeNet_DB20 --reset --quality 2
#Train_quality_30
#python main.py --template ACubeNet --save ACubeNet_DB30 --reset --quality 3
#Train_quality_40
#python main.py --template ACubeNet --save ACubeNet_DB40 --reset --quality 4

#Test_quality_10
#python main.py --template ACubeNet --data_test LIVE1+classic5 --pre_train  ../experiment/ACubeNet_DB10/model/model_best.pt --test_only --save_results --save_gt --quality 1
#Test_quality_20
#python main.py --template ACubeNet --data_test LIVE1+classic5 --pre_train  ../experiment/ACubeNet_DB20/model/model_best.pt --test_only --save_results --save_gt --quality 2
#Test_quality_30
#python main.py --template ACubeNet --data_test LIVE1+classic5 --pre_train  ../experiment/ACubeNet_DB30/model/model_best.pt --test_only --save_results --save_gt --quality 3
#Test_quality_40
#python main.py --template ACubeNet --data_test LIVE1+classic5 --pre_train  ../experiment/ACubeNet_DB40/model/model_best.pt --test_only --save_results --save_gt --quality 4

