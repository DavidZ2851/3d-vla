python data_processing/calvin_to_zarr.py \
    --root /data/group_data/katefgroup/VLA/calvin_dataset/task_ABC_D \
    --tgt /data/user_data/ngkanats/zarr_datasets/CALVIN_10th_zarr \
    --store_every 10 \
    --store_val 0 \
    --store_instructions 0

python data_processing/rechunk.py \
    --src /data/user_data/ngkanats/zarr_datasets/CALVIN_10th_zarr/train.zarr \
    --tgt /data/user_data/ngkanats/zarr_datasets/CALVIN_10th_zarr/train_rechunked8.zarr \
    --chunk_size 8 \
    --shuffle true
