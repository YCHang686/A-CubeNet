
# Train_ACubeNet_SRX2, input=48x48, output=96x96
#python main.py --template ACubeNet --save ACubeNet_SRX2 --scale 2 --reset --save_results --patch_size 96
# Train_ACubeNet_SRX3, input=48x48, output=144x144
#python main.py --template ACubeNet --save ACubeNet_SRX3 --scale 3 --reset --save_results --patch_size 144 --pre_train ../experiment/ACubeNet_SRX2/model/model_best.pt
# Train_ACubeNet_SRX4, input=48x48, output=192x192
#python main.py --template ACubeNet --save ACubeNet_SRX4 --scale 4 --reset --save_results --patch_size 192 --pre_train ../experiment/ACubeNet_SRX2/model/model_best.pt

#Test_ACubeNet_SRX2, no self-ensemble
#python main.py --template ACubeNet --data_test Set5+Set14+B100+Urban100+Manga109 --scale 2 --pre_train ../experiment/ACubeNet_SRX2/model/model_best.pt --test_only --save_results
#Test_ACubeNet_SRX3, no self-ensemble
#python main.py --template ACubeNet --data_test Set5+Set14+B100+Urban100+Manga109 --scale 3 --pre_train ../experiment/ACubeNet_SRX3/model/model_best.pt --test_only --save_results
#Test_ACubeNet_SRX4, no self-ensemble
#python main.py --template ACubeNet --data_test Set5+Set14+B100+Urban100+Manga109 --scale 4 --pre_train ../experiment/ACubeNet_SRX4/model/model_best.pt --test_only --save_results

#Test_ACubeNet_SRX2, use self-ensemble
#python main.py --template ACubeNet --data_test Set5+Set14+B100+Urban100+Manga109 --scale 2 --pre_train ../experiment/ACubeNet_SRX2/model/model_best.pt --test_only --self_ensemble --save_results
#Test_ACubeNet_SRX3, use self-ensemble
#python main.py --template ACubeNet --data_test Set5+Set14+B100+Urban100+Manga109 --scale 3 --pre_train ../experiment/ACubeNet_SRX3/model/model_best.pt --test_only --self_ensemble --save_results
#Test_ACubeNet_SRX4, use self-ensemble
#python main.py --template ACubeNet --data_test Set5+Set14+B100+Urban100+Manga109 --scale 4 --pre_train ../experiment/ACubeNet_SRX4/model/model_best.pt --test_only --self_ensemble --save_results

