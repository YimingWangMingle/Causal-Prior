cd ../..
for seed in 1;
do
	sleep 5
        python3 main.py \
          --custom_xml environments/humanoids \
          --seed $seed \
          --td \
          --bu \
          --lr 0.0001 \
          --observation_graph_type line \
          --label humanoids_smp_line&
done
cd scripts

