# ActorsHQ
python test.py --config ./config/eval_configs/actor01_face.yaml --model_dir ./logs/actorshq_avatars/Actor01 --out_dir ./logs/test_actorshq/Actor01_test --data_dir /mounted/media/dsvitov/Crucial_X6/Actor01/Sequence1/1x --test
python test.py --config ./config/eval_configs/actor02_face.yaml --model_dir ./logs/actorshq_avatars/Actor02 --out_dir ./logs/test_actorshq/Actor02_test --data_dir /mounted/media/dsvitov/Crucial_X6/Actor02/Sequence1/1x --test
python test.py --config ./config/eval_configs/actor04_face.yaml --model_dir ./logs/actorshq_avatars/Actor04 --out_dir ./logs/test_actorshq/Actor04_test --data_dir /mounted/media/dsvitov/Crucial_X6/Actor04/Sequence1/1x --test
python test.py --config ./config/eval_configs/actor05_face.yaml --model_dir ./logs/actorshq_avatars/Actor05 --out_dir ./logs/test_actorshq/Actor05_test --data_dir /mounted/media/dsvitov/Crucial_X61/Actor05/Sequence1/1x --test
python test.py --config ./config/eval_configs/actor06_face.yaml --model_dir ./logs/actorshq_avatars/Actor06 --out_dir ./logs/test_actorshq/Actor06_test --data_dir /mounted/media/dsvitov/Crucial_X61/Actor06/Sequence1/1x --test
python test.py --config ./config/eval_configs/actor07_face.yaml --model_dir ./logs/actorshq_avatars/Actor07 --out_dir ./logs/test_actorshq/Actor07_test --data_dir /mounted/media/dsvitov/Crucial_X61/Actor07/Sequence1/1x --test
python test.py --config ./config/eval_configs/actor08_face.yaml --model_dir ./logs/actorshq_avatars/Actor08 --out_dir ./logs/test_actorshq/Actor08_test --data_dir /mounted/media/dsvitov/Crucial_X61/Actor08/Sequence1/1x --test
python eval_metrics.py --data_dir ./logs/test_actorshq/

# THuman
python test.py --config ./config/subject00.yaml --data_dir /mounted/home/dsvitov/Datasets_avatars/THuman/subject00 --model_dir ./logs/thuman_avatar/subject00 --out_dir ./logs/test_thuman/subject00 --test
python eval_metrics.py --data_dir ./logs/test_thuman/
