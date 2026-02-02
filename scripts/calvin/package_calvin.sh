python data_processing/calvin_to_zarr.py \
    --root /data/group_data/katefgroup/VLA/calvin_dataset/task_ABC_D \
    --tgt /data/user_data/ngkanats/zarr_datasets/CALVIN_zarr

python data_processing/rechunk.py \
    --src /data/user_data/ngkanats/zarr_datasets/CALVIN_zarr/train.zarr \
    --tgt /data/user_data/ngkanats/zarr_datasets/CALVIN_zarr/train_rechunked8.zarr \
    --chunk_size 8 \
    --shuffle true

python data_processing/rechunk.py \
    --src /data/user_data/ngkanats/zarr_datasets/CALVIN_zarr/val.zarr \
    --tgt /data/user_data/ngkanats/zarr_datasets/CALVIN_zarr/val_rechunked8.zarr \
    --chunk_size 8 \
    --shuffle false
