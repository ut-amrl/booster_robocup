Example IsaacLab scripts using Booster T1 and algorithm based on [this](https://arxiv.org/html/2506.09588) paper.


To install packages run:
```bash
./isaaclab.sh -i
```

To run training use command:
```bash
./isaaclab.sh -p scripts/attention/train.py --task Attention-T1-v0
```

This will likely fail due to lack of resources. This model will require a multi-gpu set up to train. The multi-gpu command will look like:
```bash
CUDA_VISIBLE_DEVICES=2,3,8,9 python -m torch.distributed.run --nnodes=1 --nproc_per_node=4 scripts/student_teacher/train.py --task=Teacher-T1-v0 --headless --distributed --num_envs 1100 --video
```
Where --nproc_per_node is the number of GPU's.

Documentation on multi-gpu training can be found [here](https://isaac-sim.github.io/IsaacLab/main/source/features/multi_gpu.html)