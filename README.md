# htxt-object-detection
航天星图杯-目标检测题

```
python /d/tf-models/research/object_detection/model_main.py \
    --pipeline_config_path=./pipeline.config \
    --model_dir=./output \
    --num_train_steps=200000 \
    --num_eval_steps=50 \
    --alsologtostderr
```