#!/bin/bash
LC_NUMERIC="en_US.UTF-8"
percentages=(1 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1)
max_steps=(500 556 625 714 833 1000 1250 1667 2500 5000)
path="/content/drive/MyDrive/Financial Extension/GBM only - incremental mixing"
synthetic_path="/content/chronos-forecasting/data/GBM_synth/"
mixup_path="/content/drive/MyDrive/GeneratedData/tsmixup_data.arrow"
model_name="checkpoint-final"
highest_run=$(printf "%s\n" $path/run-* | grep -o '[0-9]\+' | sort -n | tail -n 1)
echo $highest_run
count=1
for percentage in "${percentages[@]}"; do
    complement=$(echo "1 $percentage" | awk '{printf "%.1f", $1 - $2}')
    echo "Train $count: percentage of synthetic ${percentage}, percentage of real data ${complement}"
    rm bob_configs/bob_config.yaml
    sed "s/ORIGINAL_DATA/${complement}/g" ./bob_configs/config_template.yaml > ./bob_configs/bob_config.yaml   
    sed -i "s/SYNTHETIC_DATA/${percentage}/g" ./bob_configs/bob_config.yaml
    sed -i "s|OUTPUT_DIR|$path/|g" ./bob_configs/bob_config.yaml
    sed -i "s|ORIGINAL_FILE|$mixup_path|g" ./bob_configs/bob_config.yaml
    sed -i "s|SYNTHETIC_FILE|$synthetic_path\brownian_motions_split-$count.arrow|g"  ./bob_configs/bob_config.yaml
    sed -i "s|MAX_STEPS|${max_steps[count-1]}|g"  ./bob_configs/bob_config.yaml
    sed -i "s|SAVE_STEPS|${max_steps[count-1]}|g"  ./bob_configs/bob_config.yaml
    python scripts/training/train.py --config ./bob_configs/bob_config.yaml
    run=$(($highest_run+$count))
    # echo "Run: $run"
    sed -i "s|google/t5-efficient-mini|$path/run-$run/$model_name|g" ./bob_configs/bob_config.yaml 
    count=$((count+1))
done

# Training with no mixing
echo "Training with 5000 samples without using original data..."
path="/content/drive/MyDrive/Financial Extension/GBM only - no incremental mixing"
sed "s/ORIGINAL_DATA/0/g" ./bob_configs/config_template.yaml > ./bob_configs/bob_config.yaml   
sed -i "s/SYNTHETIC_DATA/1/g" ./bob_configs/bob_config.yaml
sed -i "s|OUTPUT_DIR|$path/|g" ./bob_configs/bob_config.yaml
sed -i "s|ORIGINAL_FILE|$mixup_path|g" ./bob_configs/bob_config.yaml
sed -i "s|SYNTHETIC_FILE|$synthetic_path\brownian_motions.arrow|g"  ./bob_configs/bob_config.yaml
sed -i "s|MAX_STEPS|5000|g"  ./bob_configs/bob_config.yaml
sed -i "s|SAVE_STEPS|5000|g"  ./bob_configs/bob_config.yaml
python scripts/training/train.py --config ./bob_configs/bob_config.yaml
