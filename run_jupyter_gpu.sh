#!/bin/bash
srun -p TitanBlack --gres=gpu --exclusive --chdir=/media/hpc2_storage/ibulygin/learnable-triangulation-pytorch -w ws-02 \
jupyter notebook \
--port 6006 \
--no-browser