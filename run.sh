#!/bin/bash

# 临时文件存储输出值
OUTPUT_FILE="outputs.txt"

# 清空临时文件
> $OUTPUT_FILE

for ver in {1..30}
do
    # 获取输出值并追加到临时文件
    # python -m baseline.my_way.finetune_prune --ver $ver --gpu_id 1 --save_path ./baseline/my_way/attack/pruning/pruning_with_retrain
    # python -m baseline.trigger_set.finetune_prune  --save_path ./baseline/trigger_set/attack/finetune/small_lr --ver $ver
    python -m baseline.trigger_set.trigger_set --ver $ver --gpu_id 0
done

# 调用Python脚本将输出值写入Excel文件
python write_to_excel.py $OUTPUT_FILE
