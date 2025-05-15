### sample shell ###
###
###
### dataset=credit, preprocessing=undersampling
python3 main.py --dataset credit --inprocessing vanilla --encoder gcn --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing vanilla --encoder gat --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing vanilla --encoder sage --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing vanilla --encoder h2gcn --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
##
python3 main.py --dataset credit --inprocessing vanilla --encoder gcn --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing vanilla --encoder gat --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing vanilla --encoder sage --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing vanilla --encoder h2gcn --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
##
python3 main.py --dataset credit --inprocessing vanilla --encoder gcn --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing vanilla --encoder gat --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing vanilla --encoder sage --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing vanilla --encoder h2gcn --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
##

### dataset=bail, preprocessing=undersampling
python3 main.py --dataset bail --inprocessing vanilla --encoder gcn --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing vanilla --encoder gat --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing vanilla --encoder sage --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing vanilla --encoder h2gcn --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
##
python3 main.py --dataset bail --inprocessing vanilla --encoder gcn --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing vanilla --encoder gat --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing vanilla --encoder sage --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing vanilla --encoder h2gcn --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
##
python3 main.py --dataset bail --inprocessing vanilla --encoder gcn --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing vanilla --encoder gat --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing vanilla --encoder sage --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing vanilla --encoder h2gcn --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
##

### dataset=pokec_z, preprocessing=undersampling
python3 main.py --dataset pokec_z --inprocessing vanilla --encoder gcn --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing vanilla --encoder gat --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing vanilla --encoder sage --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing vanilla --encoder h2gcn --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_z --inprocessing vanilla --encoder gcn --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing vanilla --encoder gat --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing vanilla --encoder sage --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing vanilla --encoder h2gcn --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_z --inprocessing vanilla --encoder gcn --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing vanilla --encoder gat --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing vanilla --encoder sage --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing vanilla --encoder h2gcn --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
##

### dataset=pokec_n, preprocessing=undersampling
python3 main.py --dataset pokec_n --inprocessing vanilla --encoder gcn --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing vanilla --encoder gat --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing vanilla --encoder sage --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing vanilla --encoder h2gcn --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_n --inprocessing vanilla --encoder gcn --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing vanilla --encoder gat --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing vanilla --encoder sage --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing vanilla --encoder h2gcn --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_n --inprocessing vanilla --encoder gcn --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing vanilla --encoder gat --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing vanilla --encoder sage --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing vanilla --encoder h2gcn --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
##

### dataset=pokec_n_large, preprocessing=undersampling
python3 main.py --dataset pokec_n_large --inprocessing vanilla --encoder gcn --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing vanilla --encoder gat --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing vanilla --encoder sage --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing vanilla --encoder h2gcn --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_n_large --inprocessing vanilla --encoder gcn --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing vanilla --encoder gat --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing vanilla --encoder sage --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing vanilla --encoder h2gcn --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_n_large --inprocessing vanilla --encoder gcn --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing vanilla --encoder gat --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing vanilla --encoder sage --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing vanilla --encoder h2gcn --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
##

### dataset=pokec_z_large, preprocessing=undersampling
python3 main.py --dataset pokec_z_large --inprocessing vanilla --encoder gcn --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing vanilla --encoder gat --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing vanilla --encoder sage --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing vanilla --encoder h2gcn --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_z_large --inprocessing vanilla --encoder gcn --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing vanilla --encoder gat --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing vanilla --encoder sage --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing vanilla --encoder h2gcn --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_z_large --inprocessing vanilla --encoder gcn --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing vanilla --encoder gat --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing vanilla --encoder sage --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing vanilla --encoder h2gcn --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
##

### dataset=wikidata, preprocessing=undersampling
python3 main.py --dataset wikidata --inprocessing vanilla --encoder gcn --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset wikidata --inprocessing vanilla --encoder gat --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset wikidata --inprocessing vanilla --encoder sage --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset wikidata --inprocessing vanilla --encoder h2gcn --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
##
python3 main.py --dataset wikidata --inprocessing vanilla --encoder gcn --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset wikidata --inprocessing vanilla --encoder gat --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset wikidata --inprocessing vanilla --encoder sage --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset wikidata --inprocessing vanilla --encoder h2gcn --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
##
python3 main.py --dataset wikidata --inprocessing vanilla --encoder gcn --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset wikidata --inprocessing vanilla --encoder gat --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset wikidata --inprocessing vanilla --encoder sage --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset wikidata --inprocessing vanilla --encoder h2gcn --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
##

### dataset=dbpedia, preprocessing=undersampling
python3 main.py --dataset dbpedia --inprocessing vanilla --encoder gcn --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing vanilla --encoder gat --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing vanilla --encoder sage --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing vanilla --encoder h2gcn --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
##
python3 main.py --dataset dbpedia --inprocessing vanilla --encoder gcn --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing vanilla --encoder gat --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing vanilla --encoder sage --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing vanilla --encoder h2gcn --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
##
python3 main.py --dataset dbpedia --inprocessing vanilla --encoder gcn --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing vanilla --encoder gat --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing vanilla --encoder sage --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing vanilla --encoder h2gcn --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
##

### dataset=yago, preprocessing=undersampling
python3 main.py --dataset yago --inprocessing vanilla --encoder gcn --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing vanilla --encoder gat --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing vanilla --encoder sage --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing vanilla --encoder h2gcn --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
##
python3 main.py --dataset yago --inprocessing vanilla --encoder gcn --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing vanilla --encoder gat --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing vanilla --encoder sage --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing vanilla --encoder h2gcn --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
##
python3 main.py --dataset yago --inprocessing vanilla --encoder gcn --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing vanilla --encoder gat --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing vanilla --encoder sage --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing vanilla --encoder h2gcn --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
##
