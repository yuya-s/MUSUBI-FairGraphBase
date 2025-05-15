### sample shell ###
###
###
### dataset=credit, preprocessing=undersampling
python3 main.py --dataset credit --inprocessing fairgnn --encoder gcn --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing fairgnn --encoder gat --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing fairgnn --encoder sage --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing fairgnn --encoder h2gcn --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
##
python3 main.py --dataset credit --inprocessing fairgnn --encoder gcn --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing fairgnn --encoder gat --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing fairgnn --encoder sage --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing fairgnn --encoder h2gcn --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
##
python3 main.py --dataset credit --inprocessing fairgnn --encoder gcn --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing fairgnn --encoder gat --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing fairgnn --encoder sage --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing fairgnn --encoder h2gcn --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
##

### dataset=bail, preprocessing=undersampling
python3 main.py --dataset bail --inprocessing fairgnn --encoder gcn --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing fairgnn --encoder gat --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing fairgnn --encoder sage --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing fairgnn --encoder h2gcn --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
##
python3 main.py --dataset bail --inprocessing fairgnn --encoder gcn --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing fairgnn --encoder gat --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing fairgnn --encoder sage --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing fairgnn --encoder h2gcn --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
##
python3 main.py --dataset bail --inprocessing fairgnn --encoder gcn --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing fairgnn --encoder gat --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing fairgnn --encoder sage --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing fairgnn --encoder h2gcn --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
##

### dataset=pokec_z, preprocessing=undersampling
python3 main.py --dataset pokec_z --inprocessing fairgnn --encoder gcn --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing fairgnn --encoder gat --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing fairgnn --encoder sage --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing fairgnn --encoder h2gcn --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_z --inprocessing fairgnn --encoder gcn --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing fairgnn --encoder gat --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing fairgnn --encoder sage --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing fairgnn --encoder h2gcn --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_z --inprocessing fairgnn --encoder gcn --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing fairgnn --encoder gat --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing fairgnn --encoder sage --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing fairgnn --encoder h2gcn --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
##

### dataset=pokec_n, preprocessing=undersampling
python3 main.py --dataset pokec_n --inprocessing fairgnn --encoder gcn --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing fairgnn --encoder gat --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing fairgnn --encoder sage --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing fairgnn --encoder h2gcn --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_n --inprocessing fairgnn --encoder gcn --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing fairgnn --encoder gat --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing fairgnn --encoder sage --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing fairgnn --encoder h2gcn --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_n --inprocessing fairgnn --encoder gcn --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing fairgnn --encoder gat --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing fairgnn --encoder sage --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing fairgnn --encoder h2gcn --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
##

### dataset=pokec_n_large, preprocessing=undersampling
python3 main.py --dataset pokec_n_large --inprocessing fairgnn --encoder gcn --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing fairgnn --encoder gat --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing fairgnn --encoder sage --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing fairgnn --encoder h2gcn --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_n_large --inprocessing fairgnn --encoder gcn --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing fairgnn --encoder gat --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing fairgnn --encoder sage --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing fairgnn --encoder h2gcn --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_n_large --inprocessing fairgnn --encoder gcn --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing fairgnn --encoder gat --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing fairgnn --encoder sage --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing fairgnn --encoder h2gcn --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
##

### dataset=pokec_z_large, preprocessing=undersampling
python3 main.py --dataset pokec_z_large --inprocessing fairgnn --encoder gcn --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing fairgnn --encoder gat --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing fairgnn --encoder sage --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing fairgnn --encoder h2gcn --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_z_large --inprocessing fairgnn --encoder gcn --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing fairgnn --encoder gat --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing fairgnn --encoder sage --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing fairgnn --encoder h2gcn --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_z_large --inprocessing fairgnn --encoder gcn --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing fairgnn --encoder gat --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing fairgnn --encoder sage --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing fairgnn --encoder h2gcn --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
##

### dataset=wikidata, preprocessing=undersampling
python3 main.py --dataset wikidata --inprocessing fairgnn --encoder gcn --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset wikidata --inprocessing fairgnn --encoder gat --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset wikidata --inprocessing fairgnn --encoder sage --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset wikidata --inprocessing fairgnn --encoder h2gcn --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
##
python3 main.py --dataset wikidata --inprocessing fairgnn --encoder gcn --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset wikidata --inprocessing fairgnn --encoder gat --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset wikidata --inprocessing fairgnn --encoder sage --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset wikidata --inprocessing fairgnn --encoder h2gcn --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
##
python3 main.py --dataset wikidata --inprocessing fairgnn --encoder gcn --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset wikidata --inprocessing fairgnn --encoder gat --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset wikidata --inprocessing fairgnn --encoder sage --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset wikidata --inprocessing fairgnn --encoder h2gcn --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
##

### dataset=dbpedia, preprocessing=undersampling
python3 main.py --dataset dbpedia --inprocessing fairgnn --encoder gcn --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing fairgnn --encoder gat --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing fairgnn --encoder sage --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing fairgnn --encoder h2gcn --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
##
python3 main.py --dataset dbpedia --inprocessing fairgnn --encoder gcn --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing fairgnn --encoder gat --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing fairgnn --encoder sage --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing fairgnn --encoder h2gcn --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
##
python3 main.py --dataset dbpedia --inprocessing fairgnn --encoder gcn --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing fairgnn --encoder gat --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing fairgnn --encoder sage --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing fairgnn --encoder h2gcn --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
##

### dataset=yago, preprocessing=undersampling
python3 main.py --dataset yago --inprocessing fairgnn --encoder gcn --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing fairgnn --encoder gat --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing fairgnn --encoder sage --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing fairgnn --encoder h2gcn --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
##
python3 main.py --dataset yago --inprocessing fairgnn --encoder gcn --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing fairgnn --encoder gat --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing fairgnn --encoder sage --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing fairgnn --encoder h2gcn --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
##
python3 main.py --dataset yago --inprocessing fairgnn --encoder gcn --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing fairgnn --encoder gat --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing fairgnn --encoder sage --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing fairgnn --encoder h2gcn --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
##


### sample shell ###
###
###
### dataset=credit, preprocessing=undersampling
python3 main.py --dataset credit --inprocessing nifty --encoder gcn --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing nifty --encoder gat --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing nifty --encoder sage --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing nifty --encoder h2gcn --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
##
python3 main.py --dataset credit --inprocessing nifty --encoder gcn --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing nifty --encoder gat --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing nifty --encoder sage --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing nifty --encoder h2gcn --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
##
python3 main.py --dataset credit --inprocessing nifty --encoder gcn --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing nifty --encoder gat --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing nifty --encoder sage --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing nifty --encoder h2gcn --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
##

### dataset=bail, preprocessing=undersampling
python3 main.py --dataset bail --inprocessing nifty --encoder gcn --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing nifty --encoder gat --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing nifty --encoder sage --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing nifty --encoder h2gcn --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
##
python3 main.py --dataset bail --inprocessing nifty --encoder gcn --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing nifty --encoder gat --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing nifty --encoder sage --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing nifty --encoder h2gcn --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
##
python3 main.py --dataset bail --inprocessing nifty --encoder gcn --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing nifty --encoder gat --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing nifty --encoder sage --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing nifty --encoder h2gcn --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
##

### dataset=pokec_z, preprocessing=undersampling
python3 main.py --dataset pokec_z --inprocessing nifty --encoder gcn --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing nifty --encoder gat --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing nifty --encoder sage --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing nifty --encoder h2gcn --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_z --inprocessing nifty --encoder gcn --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing nifty --encoder gat --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing nifty --encoder sage --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing nifty --encoder h2gcn --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_z --inprocessing nifty --encoder gcn --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing nifty --encoder gat --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing nifty --encoder sage --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing nifty --encoder h2gcn --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
##

### dataset=pokec_n, preprocessing=undersampling
python3 main.py --dataset pokec_n --inprocessing nifty --encoder gcn --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing nifty --encoder gat --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing nifty --encoder sage --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing nifty --encoder h2gcn --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_n --inprocessing nifty --encoder gcn --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing nifty --encoder gat --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing nifty --encoder sage --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing nifty --encoder h2gcn --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_n --inprocessing nifty --encoder gcn --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing nifty --encoder gat --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing nifty --encoder sage --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing nifty --encoder h2gcn --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
##

### dataset=pokec_n_large, preprocessing=undersampling
python3 main.py --dataset pokec_n_large --inprocessing nifty --encoder gcn --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing nifty --encoder gat --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing nifty --encoder sage --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing nifty --encoder h2gcn --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_n_large --inprocessing nifty --encoder gcn --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing nifty --encoder gat --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing nifty --encoder sage --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing nifty --encoder h2gcn --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_n_large --inprocessing nifty --encoder gcn --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing nifty --encoder gat --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing nifty --encoder sage --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing nifty --encoder h2gcn --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
##

### dataset=pokec_z_large, preprocessing=undersampling
python3 main.py --dataset pokec_z_large --inprocessing nifty --encoder gcn --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing nifty --encoder gat --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing nifty --encoder sage --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing nifty --encoder h2gcn --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_z_large --inprocessing nifty --encoder gcn --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing nifty --encoder gat --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing nifty --encoder sage --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing nifty --encoder h2gcn --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_z_large --inprocessing nifty --encoder gcn --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing nifty --encoder gat --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing nifty --encoder sage --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing nifty --encoder h2gcn --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
##

### dataset=wikidata, preprocessing=undersampling
python3 main.py --dataset wikidata --inprocessing nifty --encoder gcn --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset wikidata --inprocessing nifty --encoder gat --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset wikidata --inprocessing nifty --encoder sage --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset wikidata --inprocessing nifty --encoder h2gcn --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
##
python3 main.py --dataset wikidata --inprocessing nifty --encoder gcn --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset wikidata --inprocessing nifty --encoder gat --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset wikidata --inprocessing nifty --encoder sage --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset wikidata --inprocessing nifty --encoder h2gcn --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
##
python3 main.py --dataset wikidata --inprocessing nifty --encoder gcn --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset wikidata --inprocessing nifty --encoder gat --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset wikidata --inprocessing nifty --encoder sage --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset wikidata --inprocessing nifty --encoder h2gcn --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
##

### dataset=dbpedia, preprocessing=undersampling
python3 main.py --dataset dbpedia --inprocessing nifty --encoder gcn --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing nifty --encoder gat --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing nifty --encoder sage --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing nifty --encoder h2gcn --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
##
python3 main.py --dataset dbpedia --inprocessing nifty --encoder gcn --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing nifty --encoder gat --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing nifty --encoder sage --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing nifty --encoder h2gcn --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
##
python3 main.py --dataset dbpedia --inprocessing nifty --encoder gcn --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing nifty --encoder gat --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing nifty --encoder sage --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing nifty --encoder h2gcn --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
##

### dataset=yago, preprocessing=undersampling
python3 main.py --dataset yago --inprocessing nifty --encoder gcn --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing nifty --encoder gat --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing nifty --encoder sage --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing nifty --encoder h2gcn --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
##
python3 main.py --dataset yago --inprocessing nifty --encoder gcn --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing nifty --encoder gat --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing nifty --encoder sage --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing nifty --encoder h2gcn --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
##
python3 main.py --dataset yago --inprocessing nifty --encoder gcn --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing nifty --encoder gat --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing nifty --encoder sage --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing nifty --encoder h2gcn --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
##


### sample shell ###
###
###
### dataset=credit, preprocessing=undersampling
python3 main.py --dataset credit --inprocessing fairsin --encoder gcn --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing fairsin --encoder gat --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing fairsin --encoder sage --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing fairsin --encoder h2gcn --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
##
python3 main.py --dataset credit --inprocessing fairsin --encoder gcn --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing fairsin --encoder gat --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing fairsin --encoder sage --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing fairsin --encoder h2gcn --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
##
python3 main.py --dataset credit --inprocessing fairsin --encoder gcn --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing fairsin --encoder gat --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing fairsin --encoder sage --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing fairsin --encoder h2gcn --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
##

### dataset=bail, preprocessing=undersampling
python3 main.py --dataset bail --inprocessing fairsin --encoder gcn --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing fairsin --encoder gat --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing fairsin --encoder sage --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing fairsin --encoder h2gcn --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
##
python3 main.py --dataset bail --inprocessing fairsin --encoder gcn --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing fairsin --encoder gat --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing fairsin --encoder sage --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing fairsin --encoder h2gcn --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
##
python3 main.py --dataset bail --inprocessing fairsin --encoder gcn --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing fairsin --encoder gat --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing fairsin --encoder sage --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing fairsin --encoder h2gcn --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
##

### dataset=pokec_z, preprocessing=undersampling
python3 main.py --dataset pokec_z --inprocessing fairsin --encoder gcn --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing fairsin --encoder gat --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing fairsin --encoder sage --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing fairsin --encoder h2gcn --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_z --inprocessing fairsin --encoder gcn --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing fairsin --encoder gat --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing fairsin --encoder sage --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing fairsin --encoder h2gcn --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_z --inprocessing fairsin --encoder gcn --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing fairsin --encoder gat --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing fairsin --encoder sage --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing fairsin --encoder h2gcn --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
##

### dataset=pokec_n, preprocessing=undersampling
python3 main.py --dataset pokec_n --inprocessing fairsin --encoder gcn --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing fairsin --encoder gat --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing fairsin --encoder sage --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing fairsin --encoder h2gcn --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_n --inprocessing fairsin --encoder gcn --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing fairsin --encoder gat --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing fairsin --encoder sage --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing fairsin --encoder h2gcn --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_n --inprocessing fairsin --encoder gcn --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing fairsin --encoder gat --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing fairsin --encoder sage --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing fairsin --encoder h2gcn --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
##

### dataset=pokec_n_large, preprocessing=undersampling
python3 main.py --dataset pokec_n_large --inprocessing fairsin --encoder gcn --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing fairsin --encoder gat --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing fairsin --encoder sage --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing fairsin --encoder h2gcn --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_n_large --inprocessing fairsin --encoder gcn --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing fairsin --encoder gat --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing fairsin --encoder sage --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing fairsin --encoder h2gcn --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_n_large --inprocessing fairsin --encoder gcn --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing fairsin --encoder gat --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing fairsin --encoder sage --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing fairsin --encoder h2gcn --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
##

### dataset=pokec_z_large, preprocessing=undersampling
python3 main.py --dataset pokec_z_large --inprocessing fairsin --encoder gcn --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing fairsin --encoder gat --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing fairsin --encoder sage --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing fairsin --encoder h2gcn --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_z_large --inprocessing fairsin --encoder gcn --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing fairsin --encoder gat --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing fairsin --encoder sage --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing fairsin --encoder h2gcn --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_z_large --inprocessing fairsin --encoder gcn --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing fairsin --encoder gat --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing fairsin --encoder sage --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing fairsin --encoder h2gcn --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
##

### dataset=wikidata, preprocessing=undersampling
python3 main.py --dataset wikidata --inprocessing fairsin --encoder gcn --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset wikidata --inprocessing fairsin --encoder gat --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset wikidata --inprocessing fairsin --encoder sage --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset wikidata --inprocessing fairsin --encoder h2gcn --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
##
python3 main.py --dataset wikidata --inprocessing fairsin --encoder gcn --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset wikidata --inprocessing fairsin --encoder gat --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset wikidata --inprocessing fairsin --encoder sage --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset wikidata --inprocessing fairsin --encoder h2gcn --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
##
python3 main.py --dataset wikidata --inprocessing fairsin --encoder gcn --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset wikidata --inprocessing fairsin --encoder gat --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset wikidata --inprocessing fairsin --encoder sage --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset wikidata --inprocessing fairsin --encoder h2gcn --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
##

### dataset=dbpedia, preprocessing=undersampling
python3 main.py --dataset dbpedia --inprocessing fairsin --encoder gcn --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing fairsin --encoder gat --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing fairsin --encoder sage --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing fairsin --encoder h2gcn --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
##
python3 main.py --dataset dbpedia --inprocessing fairsin --encoder gcn --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing fairsin --encoder gat --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing fairsin --encoder sage --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing fairsin --encoder h2gcn --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
##
python3 main.py --dataset dbpedia --inprocessing fairsin --encoder gcn --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing fairsin --encoder gat --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing fairsin --encoder sage --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing fairsin --encoder h2gcn --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
##

### dataset=yago, preprocessing=undersampling
python3 main.py --dataset yago --inprocessing fairsin --encoder gcn --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing fairsin --encoder gat --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing fairsin --encoder sage --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing fairsin --encoder h2gcn --optimize --metrics acc  --preprocessing undersampling --runs 5 --device 'cuda:4'
##
python3 main.py --dataset yago --inprocessing fairsin --encoder gcn --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing fairsin --encoder gat --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing fairsin --encoder sage --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing fairsin --encoder h2gcn --optimize --metrics f1  --preprocessing undersampling --runs 5 --device 'cuda:4'
##
python3 main.py --dataset yago --inprocessing fairsin --encoder gcn --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing fairsin --encoder gat --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing fairsin --encoder sage --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing fairsin --encoder h2gcn --optimize --metrics alpha --alpha 0.5  --preprocessing undersampling --runs 5 --device 'cuda:4'
##
