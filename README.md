# htxt-object-detection
航天星图杯-目标检测题

### 环境部署

该死的需要windows环境，先用Anaconda建立虚拟环境以防万一

**安装依赖**

```
pip install --user Cython
pip install --user contextlib2
pip install --user pillow
pip install --user lxml
pip install --user jupyter
pip install --user matplotlib

```

**下载魔改后的models以及编译好的pycocotools**

http://static.wolf-tungsten.com/htxt/tf-models.rar

http://static.wolf-tungsten.com/htxt/cocoapi-master.zip

并设置环境变量

PYTHONPATH

D:\tf-models\research;
D:\tf-models\research\slim;
D:\tf-models\research\object_detection;D:\cocoapi-master\PythonAPI\pycocotools;


### 启动训练命令

```
python /d/tf-models/research/object_detection/model_main.py \
    --pipeline_config_path=.\\pipeline.config \
    --model_dir=.\\output-20180802 \
    --num_train_steps=200000 \
    --num_eval_steps=50 \
    --alsologtostderr
```

```
python /c/高分数据比赛-高睿昊/tf-models/research/object_detection/model_main.py \
    --pipeline_config_path=.\\pipeline.config \
    --model_dir=.\\output-2018-8-2 \
    --num_train_steps=200000 \
    --num_eval_steps=50 \
    --alsologtostderr
```

```
tensorflowjs_converter \
    --input_format=tf_frozen_model \
    --output_node_names='num_detections,detection_boxes,detection_scores,detection_classes,detection_masks' \
    --saved_model_tags=serve \
    exported_model/frozen_inference_graph.pb \
    exported_model/model.ckpt
```
