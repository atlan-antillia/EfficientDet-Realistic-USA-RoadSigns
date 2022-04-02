python model_inspect.py ^
  --runmode=saved_model_infer ^
  --model_name=efficientdet-d0 ^
  --saved_model_dir=./projects/USA_RoadSigns/saved_model ^
  --min_score_thresh=0.3 ^
  --hparams="image_size=1280x720,label_map=./projects/USA_RoadSigns/label_map.yaml" ^
  --input_image=./projects/USA_RoadSigns/usa_virtual_test_dataset/*.jpg ^
  --output_image_dir=./projects/USA_RoadSigns/usa_virtual_test_datasset_outputs