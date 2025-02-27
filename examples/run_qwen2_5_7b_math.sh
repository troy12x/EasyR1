set -x

export VLLM_ATTENTION_BACKEND=XFORMERS

MODEL_PATH=silx-ai/Quasar-2.5-7B-Ultra  # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/grpo_example.yaml \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.n_gpus_per_node=4
