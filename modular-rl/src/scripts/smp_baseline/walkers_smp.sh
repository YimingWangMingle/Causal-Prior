cd ../..
for seed in 1;
do
	sleep 5
        python3 main.py \
          --custom_xml environments/walkers \
          --seed $seed \
          --td \
          --bu \
          --lr 0.0001 \
          --label walkers_smp&
done
cd scripts
