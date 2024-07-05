# Train KAN Wateramrk Baseline & my Way


1. My Way

### Train Normal Model

~~~
python -m baseline.my_way.two_step_training --ver [XXX] --gpu_id [K]
~~~
- [XXX] specifies the ver of the model
- [K] specifies gpu id you use

### Finetune or Pruning

~~~
python -m baseline.my_way.finetune_prune --ver [A] --save_path [XXX] --gpu_id [K]
~~~

- [XXX] specifies the directory of `params.json`, ex: `--save_path ./baseline/my_way/attack/finetune/large_lr`
- [A] specifies the ver of the model you want to attack
- [K] specifies gpu id you use

### Verify the model
use it when you want to verify your model or clean model through your key sample
~~~
python -m baseline.my_way.verification --ver [XXX] --clean_model
~~~
- [XXX] specifies the ver of the model
- clean_model: if use, verify on clean model, else, verify on its own watermark model

2. Signal Based Method

### Train from scratch
~~~
python -m baseline.signal_based.train_from_scratch
~~~

### Train USP
~~~
python -m baseline.signal_based.train_USP --ver [A] --gpu_id [K]
~~~
- [A] specifies the ver of the model
- [K] specifies gpu id you use

### Finetune or Pruning
~~~
python -m baseline.signal_based.finetune_prune --ver [A] --save_path [XXX] --gpu_id [K]
~~~
- [XXX] specifies the directory of `params.json`, ex: `--save_path ./baseline/signal_based/attack/finetune/large_lr`
- [A] specifies the ver of the model you want to attack
- [K] specifies gpu id you use

2. Trigger Based Method

### Train margin based
~~~
python -m baseline.trigger_set.trigger_set --ver [A] --gpu_id [K]
~~~

### Verification
~~~
python -m baseline.trigger_set.verification --clean_model --ver [A]
~~~

### Finetune or Pruning
~~~
python -m baseline.trigger_set.finetune_prune  --save_path [XXX] --ver [A]
~~~
python -m baseline.trigger_set.finetune_prune  `--save_path ./baseline/trigger_set/attack/finetune/large_lr --ver 1`

