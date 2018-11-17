#!/bin/sh

python run_and_write.py -k 4 data/higher_dimensional.csv
python run_and_write.py -k 3 data/test.csv
python run_and_write.py -k 3 data/test_noisy.csv
python run_and_write.py -k 2 data/paper.csv

for i in `seq 1 5`;
do
    java -jar elki-bundle-0.7.2-SNAPSHOT.jar -dbc.in data/higher_dimensional.csv -algorithm clustering.correlation.ORCLUS -orclus.seed $i -projectedclustering.k 4 -projectedclustering.l 2 -resulthandler ResultWriter -out elki_results/elki_higher_dimensional$i
done

for i in `seq 1 5`;
do
    java -jar elki-bundle-0.7.2-SNAPSHOT.jar -dbc.in data/test.csv -algorithm clustering.correlation.ORCLUS -projectedclustering.k 3 -projectedclustering.l 2 -orclus.seed $i -resulthandler ResultWriter -out elki_results/elki_test$i
done

for i in `seq 1 5`;
do
    java -jar elki-bundle-0.7.2-SNAPSHOT.jar -dbc.in data/test_noisy.csv -algorithm clustering.correlation.ORCLUS -projectedclustering.k 3 -projectedclustering.l 2 -orclus.seed $i -resulthandler ResultWriter -out elki_results/elki_test_noisy$i
done

for i in `seq 1 5`;
do
    java -jar elki-bundle-0.7.2-SNAPSHOT.jar -dbc.in data/paper.csv -algorithm clustering.correlation.ORCLUS -projectedclustering.k 2 -projectedclustering.l 2 -orclus.seed $i -resulthandler ResultWriter -out elki_results/elki_paper$i
done
