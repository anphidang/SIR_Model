#!/bin/bash
COMMON="manager.task_names=[TurnOffSinkFaucet] manager.batch_size=16 trainer.epochs=50 trainer.test_bool=True trainer.eval_n_times=50 trainer.store_videos=False"

for SEED in 42 43; do
  echo "===== SIR seed=$SEED ====="
  python main.py $COMMON trainer.seed=$SEED method/graph_encoder=xai_gnn \
    wandb.run_name=sir_seed$SEED || echo "!!! FEHLGESCHLAGEN: sir seed=$SEED"

  echo "===== FCG seed=$SEED ====="
  python main.py $COMMON trainer.seed=$SEED method/graph_encoder=gnn \
    wandb.run_name=fcg_seed$SEED || echo "!!! FEHLGESCHLAGEN: fcg seed=$SEED"

  echo "===== IMAGE seed=$SEED ====="
  python main.py $COMMON trainer.seed=$SEED \
    manager.image_modalities=[left,right] manager.graph_modalities=[] \
    wandb.run_name=image_seed$SEED || echo "!!! FEHLGESCHLAGEN: image seed=$SEED"
done
echo "===== ALLE LAEUFE DURCH ====="
