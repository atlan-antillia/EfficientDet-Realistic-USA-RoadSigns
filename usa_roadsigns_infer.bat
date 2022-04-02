python model_inspect.py ^
  --runmode=infer ^
  --model_name=efficientdet-d0 ^
  --ckpt_path=./projects/USA_RoadSigns/model ^
  --min_score_thresh=0.3 ^
  --hparams="label_map=./projects/USA_RoadSigns/label_map.yaml" ^
  --input_image=./projects/USA_RoadSigns/test/*.jpg ^
  --output_image_dir=./projects/USA_RoadSigns/usa_virtual_test_dataset_outputs