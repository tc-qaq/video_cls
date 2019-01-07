# video_cls



cnn for video classification
run step:
# 1 data preprocessing
run data/1_move_files.py
      data/2_extract_files.py
# 2 test Incept v3 in data
run CNN_validate_images.py
# 3 train model
run CNN_train_UCF101.py
# 4 test
run CNN_evaluate_testset.py
