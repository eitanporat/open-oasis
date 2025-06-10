python generate.py \
  --oasis-ckpt /root/.cache/huggingface/hub/models--Etched--oasis-500m/snapshots/4ca7d2d811f4f0c6fd1d5719bf83f14af3446c0c/oasis500m.safetensors \
  --vae-ckpt /root/.cache/huggingface/hub/models--Etched--oasis-500m/snapshots/4ca7d2d811f4f0c6fd1d5719bf83f14af3446c0c/vit-l-20.safetensors \
  --prompt-path sample_data/sample_image_0.png \
  --actions-path sample_data/sample_actions_0.one_hot_actions.pt