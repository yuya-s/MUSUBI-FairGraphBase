### sample shell ###
###
###
### dataset=pokec_n
# python3 main.py --dataset pokec_n --inprocessing vanilla --encoder gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
# python3 main.py --dataset pokec_n --inprocessing vanilla --encoder gat --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
# python3 main.py --dataset pokec_n --inprocessing vanilla --encoder sage --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
# python3 main.py --dataset pokec_n --inprocessing vanilla --encoder h2gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
# ##
# python3 main.py --dataset pokec_n --inprocessing fairgnn --encoder gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
# python3 main.py --dataset pokec_n --inprocessing fairgnn --encoder gat --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
# python3 main.py --dataset pokec_n --inprocessing fairgnn --encoder sage --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
# python3 main.py --dataset pokec_n --inprocessing fairgnn --encoder h2gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
# ##
# python3 main.py --dataset pokec_n --inprocessing nifty --encoder gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
# python3 main.py --dataset pokec_n --inprocessing nifty --encoder gat --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
# python3 main.py --dataset pokec_n --inprocessing nifty --encoder sage --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
# python3 main.py --dataset pokec_n --inprocessing nifty --encoder h2gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
# ##
# python3 main.py --dataset pokec_n --inprocessing fairsin --encoder gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
# python3 main.py --dataset pokec_n --inprocessing fairsin --encoder gat --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
# python3 main.py --dataset pokec_n --inprocessing fairsin --encoder sage --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
# python3 main.py --dataset pokec_n --inprocessing fairsin --encoder h2gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
# ##
# python3 main.py --dataset pokec_n --inprocessing fairdrop --encoder gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
# python3 main.py --dataset pokec_n --inprocessing fairdrop --encoder gat --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
# python3 main.py --dataset pokec_n --inprocessing fairdrop --encoder sage --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
# python3 main.py --dataset pokec_n --inprocessing fairdrop --encoder h2gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'

# python3 main.py --dataset pokec_n --inprocessing vanilla --encoder gcn --optimize --metrics acc --preprocessing undersampling --runs 5 --device 'cuda:4'
# python3 main.py --dataset pokec_n --inprocessing vanilla --encoder gat --optimize --metrics acc --preprocessing undersampling --runs 5 --device 'cuda:4'
# python3 main.py --dataset pokec_n --inprocessing vanilla --encoder sage --optimize --metrics acc --preprocessing undersampling --runs 5 --device 'cuda:4'
# python3 main.py --dataset pokec_n --inprocessing vanilla --encoder h2gcn --optimize --metrics acc --preprocessing undersampling --runs 5 --device 'cuda:4'


python3 main.py --dataset pokec_n --inprocessing vanilla --encoder gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing vanilla --encoder gat --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing vanilla --encoder sage --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing vanilla --encoder h2gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_n --inprocessing fairgnn --encoder gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing fairgnn --encoder gat --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing fairgnn --encoder sage --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing fairgnn --encoder h2gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_n --inprocessing nifty --encoder gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing nifty --encoder gat --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing nifty --encoder sage --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing nifty --encoder h2gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_n --inprocessing fairsin --encoder gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing fairsin --encoder gat --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing fairsin --encoder sage --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing fairsin --encoder h2gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_n --inprocessing fairdrop --encoder gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing fairdrop --encoder gat --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing fairdrop --encoder sage --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing fairdrop --encoder h2gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'

python3 main.py --dataset pokec_n --inprocessing vanilla --encoder gcn --optimize --metrics alpha --alpha 0.5 --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing vanilla --encoder gat --optimize --metrics alpha --alpha 0.5 --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing vanilla --encoder sage --optimize --metrics alpha --alpha 0.5 --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing vanilla --encoder h2gcn --optimize --metrics alpha --alpha 0.5 --preprocessing undersampling --runs 5 --device 'cuda:4'


python3 main.py --dataset pokec_n --inprocessing vanilla --encoder gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing vanilla --encoder gat --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing vanilla --encoder sage --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing vanilla --encoder h2gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_n --inprocessing fairgnn --encoder gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing fairgnn --encoder gat --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing fairgnn --encoder sage --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing fairgnn --encoder h2gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_n --inprocessing nifty --encoder gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing nifty --encoder gat --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing nifty --encoder sage --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing nifty --encoder h2gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_n --inprocessing fairsin --encoder gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing fairsin --encoder gat --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing fairsin --encoder sage --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing fairsin --encoder h2gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_n --inprocessing fairdrop --encoder gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing fairdrop --encoder gat --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing fairdrop --encoder sage --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing fairdrop --encoder h2gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'

python3 main.py --dataset pokec_n --inprocessing vanilla --encoder gcn --optimize --metrics f1 --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing vanilla --encoder gat --optimize --metrics f1 --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing vanilla --encoder sage --optimize --metrics f1 --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing vanilla --encoder h2gcn --optimize --metrics f1 --preprocessing undersampling --runs 5 --device 'cuda:4'

###
### dataset=pokec_z
python3 main.py --dataset pokec_z --inprocessing vanilla --encoder gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing vanilla --encoder gat --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing vanilla --encoder sage --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing vanilla --encoder h2gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_z --inprocessing fairgnn --encoder gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing fairgnn --encoder gat --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing fairgnn --encoder sage --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing fairgnn --encoder h2gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_z --inprocessing nifty --encoder gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing nifty --encoder gat --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing nifty --encoder sage --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing nifty --encoder h2gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_z --inprocessing fairsin --encoder gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing fairsin --encoder gat --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing fairsin --encoder sage --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing fairsin --encoder h2gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_z --inprocessing fairdrop --encoder gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing fairdrop --encoder gat --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing fairdrop --encoder sage --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing fairdrop --encoder h2gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'

python3 main.py --dataset pokec_z --inprocessing vanilla --encoder gcn --optimize --metrics acc --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing vanilla --encoder gat --optimize --metrics acc --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing vanilla --encoder sage --optimize --metrics acc --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing vanilla --encoder h2gcn --optimize --metrics acc --preprocessing undersampling --runs 5 --device 'cuda:4'


python3 main.py --dataset pokec_z --inprocessing vanilla --encoder gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing vanilla --encoder gat --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing vanilla --encoder sage --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing vanilla --encoder h2gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_z --inprocessing fairgnn --encoder gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing fairgnn --encoder gat --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing fairgnn --encoder sage --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing fairgnn --encoder h2gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_z --inprocessing nifty --encoder gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing nifty --encoder gat --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing nifty --encoder sage --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing nifty --encoder h2gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_z --inprocessing fairsin --encoder gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing fairsin --encoder gat --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing fairsin --encoder sage --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing fairsin --encoder h2gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_z --inprocessing fairdrop --encoder gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing fairdrop --encoder gat --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing fairdrop --encoder sage --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing fairdrop --encoder h2gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'

python3 main.py --dataset pokec_z --inprocessing vanilla --encoder gcn --optimize --metrics alpha --alpha 0.5 --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing vanilla --encoder gat --optimize --metrics alpha --alpha 0.5 --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing vanilla --encoder sage --optimize --metrics alpha --alpha 0.5 --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing vanilla --encoder h2gcn --optimize --metrics alpha --alpha 0.5 --preprocessing undersampling --runs 5 --device 'cuda:4'

python3 main.py --dataset pokec_z --inprocessing vanilla --encoder gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing vanilla --encoder gat --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing vanilla --encoder sage --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing vanilla --encoder h2gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_z --inprocessing fairgnn --encoder gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing fairgnn --encoder gat --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing fairgnn --encoder sage --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing fairgnn --encoder h2gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_z --inprocessing nifty --encoder gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing nifty --encoder gat --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing nifty --encoder sage --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing nifty --encoder h2gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_z --inprocessing fairsin --encoder gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing fairsin --encoder gat --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing fairsin --encoder sage --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing fairsin --encoder h2gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_z --inprocessing fairdrop --encoder gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing fairdrop --encoder gat --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing fairdrop --encoder sage --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing fairdrop --encoder h2gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'

python3 main.py --dataset pokec_z --inprocessing vanilla --encoder gcn --optimize --metrics f1 --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing vanilla --encoder gat --optimize --metrics f1 --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing vanilla --encoder sage --optimize --metrics f1 --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing vanilla --encoder h2gcn --optimize --metrics f1 --preprocessing undersampling --runs 5 --device 'cuda:4'




### dataset=credit
python3 main.py --dataset credit --inprocessing vanilla --encoder gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing vanilla --encoder gat --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing vanilla --encoder sage --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing vanilla --encoder h2gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset credit --inprocessing fairgnn --encoder gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing fairgnn --encoder gat --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing fairgnn --encoder sage --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing fairgnn --encoder h2gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset credit --inprocessing nifty --encoder gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing nifty --encoder gat --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing nifty --encoder sage --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing nifty --encoder h2gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset credit --inprocessing fairsin --encoder gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing fairsin --encoder gat --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing fairsin --encoder sage --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing fairsin --encoder h2gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset credit --inprocessing fairdrop --encoder gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing fairdrop --encoder gat --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing fairdrop --encoder sage --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing fairdrop --encoder h2gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'

python3 main.py --dataset credit --inprocessing vanilla --encoder gcn --optimize --metrics acc --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing vanilla --encoder gat --optimize --metrics acc --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing vanilla --encoder sage --optimize --metrics acc --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing vanilla --encoder h2gcn --optimize --metrics acc --preprocessing undersampling --runs 5 --device 'cuda:4'


python3 main.py --dataset credit --inprocessing vanilla --encoder gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing vanilla --encoder gat --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing vanilla --encoder sage --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing vanilla --encoder h2gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset credit --inprocessing fairgnn --encoder gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing fairgnn --encoder gat --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing fairgnn --encoder sage --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing fairgnn --encoder h2gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset credit --inprocessing nifty --encoder gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing nifty --encoder gat --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing nifty --encoder sage --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing nifty --encoder h2gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset credit --inprocessing fairsin --encoder gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing fairsin --encoder gat --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing fairsin --encoder sage --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing fairsin --encoder h2gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset credit --inprocessing fairdrop --encoder gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing fairdrop --encoder gat --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing fairdrop --encoder sage --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing fairdrop --encoder h2gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'

python3 main.py --dataset credit --inprocessing vanilla --encoder gcn --optimize --metrics alpha --alpha 0.5 --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing vanilla --encoder gat --optimize --metrics alpha --alpha 0.5 --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing vanilla --encoder sage --optimize --metrics alpha --alpha 0.5 --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing vanilla --encoder h2gcn --optimize --metrics alpha --alpha 0.5 --preprocessing undersampling --runs 5 --device 'cuda:4'

python3 main.py --dataset credit --inprocessing vanilla --encoder gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing vanilla --encoder gat --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing vanilla --encoder sage --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing vanilla --encoder h2gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset credit --inprocessing fairgnn --encoder gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing fairgnn --encoder gat --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing fairgnn --encoder sage --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing fairgnn --encoder h2gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset credit --inprocessing nifty --encoder gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing nifty --encoder gat --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing nifty --encoder sage --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing nifty --encoder h2gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset credit --inprocessing fairsin --encoder gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing fairsin --encoder gat --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing fairsin --encoder sage --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing fairsin --encoder h2gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset credit --inprocessing fairdrop --encoder gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing fairdrop --encoder gat --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing fairdrop --encoder sage --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing fairdrop --encoder h2gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'

python3 main.py --dataset credit --inprocessing vanilla --encoder gcn --optimize --metrics f1 --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing vanilla --encoder gat --optimize --metrics f1 --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing vanilla --encoder sage --optimize --metrics f1 --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing vanilla --encoder h2gcn --optimize --metrics f1 --preprocessing undersampling --runs 5 --device 'cuda:4'

### dataset=credit
python3 main.py --dataset credit --inprocessing vanilla --encoder gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing vanilla --encoder gat --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing vanilla --encoder sage --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing vanilla --encoder h2gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset credit --inprocessing fairgnn --encoder gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing fairgnn --encoder gat --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing fairgnn --encoder sage --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing fairgnn --encoder h2gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset credit --inprocessing nifty --encoder gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing nifty --encoder gat --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing nifty --encoder sage --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing nifty --encoder h2gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset credit --inprocessing fairsin --encoder gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing fairsin --encoder gat --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing fairsin --encoder sage --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing fairsin --encoder h2gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset credit --inprocessing fairdrop --encoder gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing fairdrop --encoder gat --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing fairdrop --encoder sage --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing fairdrop --encoder h2gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'

python3 main.py --dataset credit --inprocessing vanilla --encoder gcn --optimize --metrics acc --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing vanilla --encoder gat --optimize --metrics acc --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing vanilla --encoder sage --optimize --metrics acc --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing vanilla --encoder h2gcn --optimize --metrics acc --preprocessing undersampling --runs 5 --device 'cuda:4'


python3 main.py --dataset credit --inprocessing vanilla --encoder gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing vanilla --encoder gat --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing vanilla --encoder sage --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing vanilla --encoder h2gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset credit --inprocessing fairgnn --encoder gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing fairgnn --encoder gat --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing fairgnn --encoder sage --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing fairgnn --encoder h2gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset credit --inprocessing nifty --encoder gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing nifty --encoder gat --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing nifty --encoder sage --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing nifty --encoder h2gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset credit --inprocessing fairsin --encoder gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing fairsin --encoder gat --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing fairsin --encoder sage --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing fairsin --encoder h2gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset credit --inprocessing fairdrop --encoder gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing fairdrop --encoder gat --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing fairdrop --encoder sage --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing fairdrop --encoder h2gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'

python3 main.py --dataset credit --inprocessing vanilla --encoder gcn --optimize --metrics alpha --alpha 0.5 --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing vanilla --encoder gat --optimize --metrics alpha --alpha 0.5 --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing vanilla --encoder sage --optimize --metrics alpha --alpha 0.5 --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing vanilla --encoder h2gcn --optimize --metrics alpha --alpha 0.5 --preprocessing undersampling --runs 5 --device 'cuda:4'

python3 main.py --dataset credit --inprocessing vanilla --encoder gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing vanilla --encoder gat --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing vanilla --encoder sage --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing vanilla --encoder h2gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset credit --inprocessing fairgnn --encoder gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing fairgnn --encoder gat --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing fairgnn --encoder sage --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing fairgnn --encoder h2gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset credit --inprocessing nifty --encoder gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing nifty --encoder gat --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing nifty --encoder sage --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing nifty --encoder h2gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset credit --inprocessing fairsin --encoder gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing fairsin --encoder gat --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing fairsin --encoder sage --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing fairsin --encoder h2gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset credit --inprocessing fairdrop --encoder gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing fairdrop --encoder gat --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing fairdrop --encoder sage --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing fairdrop --encoder h2gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'

python3 main.py --dataset credit --inprocessing vanilla --encoder gcn --optimize --metrics f1 --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing vanilla --encoder gat --optimize --metrics f1 --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing vanilla --encoder sage --optimize --metrics f1 --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset credit --inprocessing vanilla --encoder h2gcn --optimize --metrics f1 --preprocessing undersampling --runs 5 --device 'cuda:4'


### dataset=bail
python3 main.py --dataset bail --inprocessing vanilla --encoder gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing vanilla --encoder gat --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing vanilla --encoder sage --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing vanilla --encoder h2gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset bail --inprocessing fairgnn --encoder gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing fairgnn --encoder gat --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing fairgnn --encoder sage --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing fairgnn --encoder h2gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset bail --inprocessing nifty --encoder gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing nifty --encoder gat --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing nifty --encoder sage --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing nifty --encoder h2gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset bail --inprocessing fairsin --encoder gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing fairsin --encoder gat --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing fairsin --encoder sage --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing fairsin --encoder h2gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset bail --inprocessing fairdrop --encoder gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing fairdrop --encoder gat --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing fairdrop --encoder sage --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing fairdrop --encoder h2gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'

python3 main.py --dataset bail --inprocessing vanilla --encoder gcn --optimize --metrics acc --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing vanilla --encoder gat --optimize --metrics acc --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing vanilla --encoder sage --optimize --metrics acc --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing vanilla --encoder h2gcn --optimize --metrics acc --preprocessing undersampling --runs 5 --device 'cuda:4'


python3 main.py --dataset bail --inprocessing vanilla --encoder gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing vanilla --encoder gat --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing vanilla --encoder sage --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing vanilla --encoder h2gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset bail --inprocessing fairgnn --encoder gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing fairgnn --encoder gat --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing fairgnn --encoder sage --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing fairgnn --encoder h2gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset bail --inprocessing nifty --encoder gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing nifty --encoder gat --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing nifty --encoder sage --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing nifty --encoder h2gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset bail --inprocessing fairsin --encoder gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing fairsin --encoder gat --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing fairsin --encoder sage --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing fairsin --encoder h2gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset bail --inprocessing fairdrop --encoder gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing fairdrop --encoder gat --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing fairdrop --encoder sage --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing fairdrop --encoder h2gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'

python3 main.py --dataset bail --inprocessing vanilla --encoder gcn --optimize --metrics alpha --alpha 0.5 --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing vanilla --encoder gat --optimize --metrics alpha --alpha 0.5 --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing vanilla --encoder sage --optimize --metrics alpha --alpha 0.5 --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing vanilla --encoder h2gcn --optimize --metrics alpha --alpha 0.5 --preprocessing undersampling --runs 5 --device 'cuda:4'

python3 main.py --dataset bail --inprocessing vanilla --encoder gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing vanilla --encoder gat --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing vanilla --encoder sage --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing vanilla --encoder h2gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset bail --inprocessing fairgnn --encoder gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing fairgnn --encoder gat --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing fairgnn --encoder sage --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing fairgnn --encoder h2gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset bail --inprocessing nifty --encoder gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing nifty --encoder gat --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing nifty --encoder sage --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing nifty --encoder h2gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset bail --inprocessing fairsin --encoder gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing fairsin --encoder gat --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing fairsin --encoder sage --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing fairsin --encoder h2gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset bail --inprocessing fairdrop --encoder gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing fairdrop --encoder gat --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing fairdrop --encoder sage --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing fairdrop --encoder h2gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'

python3 main.py --dataset bail --inprocessing vanilla --encoder gcn --optimize --metrics f1 --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing vanilla --encoder gat --optimize --metrics f1 --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing vanilla --encoder sage --optimize --metrics f1 --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset bail --inprocessing vanilla --encoder h2gcn --optimize --metrics f1 --preprocessing undersampling --runs 5 --device 'cuda:4'


### dataset=yago
python3 main.py --dataset yago --inprocessing vanilla --encoder gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing vanilla --encoder gat --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing vanilla --encoder sage --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing vanilla --encoder h2gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset yago --inprocessing fairgnn --encoder gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing fairgnn --encoder gat --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing fairgnn --encoder sage --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing fairgnn --encoder h2gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset yago --inprocessing nifty --encoder gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing nifty --encoder gat --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing nifty --encoder sage --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing nifty --encoder h2gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset yago --inprocessing fairsin --encoder gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing fairsin --encoder gat --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing fairsin --encoder sage --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing fairsin --encoder h2gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset yago --inprocessing fairdrop --encoder gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing fairdrop --encoder gat --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing fairdrop --encoder sage --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing fairdrop --encoder h2gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'

python3 main.py --dataset yago --inprocessing vanilla --encoder gcn --optimize --metrics acc --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing vanilla --encoder gat --optimize --metrics acc --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing vanilla --encoder sage --optimize --metrics acc --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing vanilla --encoder h2gcn --optimize --metrics acc --preprocessing undersampling --runs 5 --device 'cuda:4'


python3 main.py --dataset yago --inprocessing vanilla --encoder gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing vanilla --encoder gat --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing vanilla --encoder sage --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing vanilla --encoder h2gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset yago --inprocessing fairgnn --encoder gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing fairgnn --encoder gat --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing fairgnn --encoder sage --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing fairgnn --encoder h2gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset yago --inprocessing nifty --encoder gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing nifty --encoder gat --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing nifty --encoder sage --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing nifty --encoder h2gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset yago --inprocessing fairsin --encoder gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing fairsin --encoder gat --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing fairsin --encoder sage --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing fairsin --encoder h2gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset yago --inprocessing fairdrop --encoder gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing fairdrop --encoder gat --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing fairdrop --encoder sage --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing fairdrop --encoder h2gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'

python3 main.py --dataset yago --inprocessing vanilla --encoder gcn --optimize --metrics alpha --alpha 0.5 --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing vanilla --encoder gat --optimize --metrics alpha --alpha 0.5 --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing vanilla --encoder sage --optimize --metrics alpha --alpha 0.5 --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing vanilla --encoder h2gcn --optimize --metrics alpha --alpha 0.5 --preprocessing undersampling --runs 5 --device 'cuda:4'

python3 main.py --dataset yago --inprocessing vanilla --encoder gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing vanilla --encoder gat --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing vanilla --encoder sage --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing vanilla --encoder h2gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset yago --inprocessing fairgnn --encoder gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing fairgnn --encoder gat --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing fairgnn --encoder sage --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing fairgnn --encoder h2gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset yago --inprocessing nifty --encoder gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing nifty --encoder gat --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing nifty --encoder sage --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing nifty --encoder h2gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset yago --inprocessing fairsin --encoder gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing fairsin --encoder gat --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing fairsin --encoder sage --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing fairsin --encoder h2gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset yago --inprocessing fairdrop --encoder gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing fairdrop --encoder gat --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing fairdrop --encoder sage --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing fairdrop --encoder h2gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'

python3 main.py --dataset yago --inprocessing vanilla --encoder gcn --optimize --metrics f1 --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing vanilla --encoder gat --optimize --metrics f1 --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing vanilla --encoder sage --optimize --metrics f1 --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset yago --inprocessing vanilla --encoder h2gcn --optimize --metrics f1 --preprocessing undersampling --runs 5 --device 'cuda:4'


### dataset=dbpedia
python3 main.py --dataset dbpedia --inprocessing vanilla --encoder gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing vanilla --encoder gat --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing vanilla --encoder sage --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing vanilla --encoder h2gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset dbpedia --inprocessing fairgnn --encoder gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing fairgnn --encoder gat --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing fairgnn --encoder sage --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing fairgnn --encoder h2gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset dbpedia --inprocessing nifty --encoder gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing nifty --encoder gat --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing nifty --encoder sage --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing nifty --encoder h2gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset dbpedia --inprocessing fairsin --encoder gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing fairsin --encoder gat --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing fairsin --encoder sage --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing fairsin --encoder h2gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset dbpedia --inprocessing fairdrop --encoder gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing fairdrop --encoder gat --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing fairdrop --encoder sage --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing fairdrop --encoder h2gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'

python3 main.py --dataset dbpedia --inprocessing vanilla --encoder gcn --optimize --metrics acc --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing vanilla --encoder gat --optimize --metrics acc --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing vanilla --encoder sage --optimize --metrics acc --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing vanilla --encoder h2gcn --optimize --metrics acc --preprocessing undersampling --runs 5 --device 'cuda:4'


python3 main.py --dataset dbpedia --inprocessing vanilla --encoder gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing vanilla --encoder gat --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing vanilla --encoder sage --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing vanilla --encoder h2gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset dbpedia --inprocessing fairgnn --encoder gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing fairgnn --encoder gat --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing fairgnn --encoder sage --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing fairgnn --encoder h2gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset dbpedia --inprocessing nifty --encoder gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing nifty --encoder gat --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing nifty --encoder sage --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing nifty --encoder h2gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset dbpedia --inprocessing fairsin --encoder gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing fairsin --encoder gat --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing fairsin --encoder sage --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing fairsin --encoder h2gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset dbpedia --inprocessing fairdrop --encoder gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing fairdrop --encoder gat --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing fairdrop --encoder sage --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing fairdrop --encoder h2gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'

python3 main.py --dataset dbpedia --inprocessing vanilla --encoder gcn --optimize --metrics alpha --alpha 0.5 --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing vanilla --encoder gat --optimize --metrics alpha --alpha 0.5 --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing vanilla --encoder sage --optimize --metrics alpha --alpha 0.5 --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing vanilla --encoder h2gcn --optimize --metrics alpha --alpha 0.5 --preprocessing undersampling --runs 5 --device 'cuda:4'

python3 main.py --dataset dbpedia --inprocessing vanilla --encoder gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing vanilla --encoder gat --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing vanilla --encoder sage --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing vanilla --encoder h2gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset dbpedia --inprocessing fairgnn --encoder gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing fairgnn --encoder gat --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing fairgnn --encoder sage --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing fairgnn --encoder h2gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset dbpedia --inprocessing nifty --encoder gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing nifty --encoder gat --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing nifty --encoder sage --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing nifty --encoder h2gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset dbpedia --inprocessing fairsin --encoder gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing fairsin --encoder gat --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing fairsin --encoder sage --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing fairsin --encoder h2gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset dbpedia --inprocessing fairdrop --encoder gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing fairdrop --encoder gat --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing fairdrop --encoder sage --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing fairdrop --encoder h2gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'

python3 main.py --dataset dbpedia --inprocessing vanilla --encoder gcn --optimize --metrics f1 --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing vanilla --encoder gat --optimize --metrics f1 --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing vanilla --encoder sage --optimize --metrics f1 --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset dbpedia --inprocessing vanilla --encoder h2gcn --optimize --metrics f1 --preprocessing undersampling --runs 5 --device 'cuda:4'


### dataset=pokec_n_large
python3 main.py --dataset pokec_n_large --inprocessing vanilla --encoder gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing vanilla --encoder gat --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing vanilla --encoder sage --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing vanilla --encoder h2gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_n_large --inprocessing fairgnn --encoder gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing fairgnn --encoder gat --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing fairgnn --encoder sage --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing fairgnn --encoder h2gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_n_large --inprocessing nifty --encoder gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing nifty --encoder gat --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing nifty --encoder sage --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing nifty --encoder h2gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_n_large --inprocessing fairsin --encoder gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing fairsin --encoder gat --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing fairsin --encoder sage --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing fairsin --encoder h2gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_n_large --inprocessing fairdrop --encoder gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing fairdrop --encoder gat --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing fairdrop --encoder sage --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing fairdrop --encoder h2gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'

python3 main.py --dataset pokec_n_large --inprocessing vanilla --encoder gcn --optimize --metrics acc --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing vanilla --encoder gat --optimize --metrics acc --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing vanilla --encoder sage --optimize --metrics acc --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing vanilla --encoder h2gcn --optimize --metrics acc --preprocessing undersampling --runs 5 --device 'cuda:4'


python3 main.py --dataset pokec_n_large --inprocessing vanilla --encoder gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing vanilla --encoder gat --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing vanilla --encoder sage --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing vanilla --encoder h2gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_n_large --inprocessing fairgnn --encoder gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing fairgnn --encoder gat --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing fairgnn --encoder sage --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing fairgnn --encoder h2gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_n_large --inprocessing nifty --encoder gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing nifty --encoder gat --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing nifty --encoder sage --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing nifty --encoder h2gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_n_large --inprocessing fairsin --encoder gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing fairsin --encoder gat --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing fairsin --encoder sage --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing fairsin --encoder h2gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_n_large --inprocessing fairdrop --encoder gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing fairdrop --encoder gat --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing fairdrop --encoder sage --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing fairdrop --encoder h2gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'

python3 main.py --dataset pokec_n_large --inprocessing vanilla --encoder gcn --optimize --metrics alpha --alpha 0.5 --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing vanilla --encoder gat --optimize --metrics alpha --alpha 0.5 --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing vanilla --encoder sage --optimize --metrics alpha --alpha 0.5 --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing vanilla --encoder h2gcn --optimize --metrics alpha --alpha 0.5 --preprocessing undersampling --runs 5 --device 'cuda:4'

python3 main.py --dataset pokec_n_large --inprocessing vanilla --encoder gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing vanilla --encoder gat --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing vanilla --encoder sage --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing vanilla --encoder h2gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_n_large --inprocessing fairgnn --encoder gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing fairgnn --encoder gat --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing fairgnn --encoder sage --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing fairgnn --encoder h2gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_n_large --inprocessing nifty --encoder gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing nifty --encoder gat --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing nifty --encoder sage --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing nifty --encoder h2gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_n_large --inprocessing fairsin --encoder gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing fairsin --encoder gat --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing fairsin --encoder sage --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing fairsin --encoder h2gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_n_large --inprocessing fairdrop --encoder gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing fairdrop --encoder gat --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing fairdrop --encoder sage --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing fairdrop --encoder h2gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'

python3 main.py --dataset pokec_n_large --inprocessing vanilla --encoder gcn --optimize --metrics f1 --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing vanilla --encoder gat --optimize --metrics f1 --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing vanilla --encoder sage --optimize --metrics f1 --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n_large --inprocessing vanilla --encoder h2gcn --optimize --metrics f1 --preprocessing undersampling --runs 5 --device 'cuda:4'



### dataset=pokec_z_large
python3 main.py --dataset pokec_z_large --inprocessing vanilla --encoder gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing vanilla --encoder gat --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing vanilla --encoder sage --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing vanilla --encoder h2gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_z_large --inprocessing fairgnn --encoder gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing fairgnn --encoder gat --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing fairgnn --encoder sage --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing fairgnn --encoder h2gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_z_large --inprocessing nifty --encoder gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing nifty --encoder gat --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing nifty --encoder sage --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing nifty --encoder h2gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_z_large --inprocessing fairsin --encoder gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing fairsin --encoder gat --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing fairsin --encoder sage --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing fairsin --encoder h2gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_z_large --inprocessing fairdrop --encoder gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing fairdrop --encoder gat --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing fairdrop --encoder sage --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing fairdrop --encoder h2gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'

python3 main.py --dataset pokec_z_large --inprocessing vanilla --encoder gcn --optimize --metrics acc --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing vanilla --encoder gat --optimize --metrics acc --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing vanilla --encoder sage --optimize --metrics acc --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing vanilla --encoder h2gcn --optimize --metrics acc --preprocessing undersampling --runs 5 --device 'cuda:4'


python3 main.py --dataset pokec_z_large --inprocessing vanilla --encoder gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing vanilla --encoder gat --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing vanilla --encoder sage --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing vanilla --encoder h2gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_z_large --inprocessing fairgnn --encoder gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing fairgnn --encoder gat --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing fairgnn --encoder sage --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing fairgnn --encoder h2gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_z_large --inprocessing nifty --encoder gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing nifty --encoder gat --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing nifty --encoder sage --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing nifty --encoder h2gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_z_large --inprocessing fairsin --encoder gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing fairsin --encoder gat --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing fairsin --encoder sage --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing fairsin --encoder h2gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_z_large --inprocessing fairdrop --encoder gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing fairdrop --encoder gat --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing fairdrop --encoder sage --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing fairdrop --encoder h2gcn --optimize --metrics alpha --alpha 0.5  --preprocessing None --runs 5 --device 'cuda:4'

python3 main.py --dataset pokec_z_large --inprocessing vanilla --encoder gcn --optimize --metrics alpha --alpha 0.5 --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing vanilla --encoder gat --optimize --metrics alpha --alpha 0.5 --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing vanilla --encoder sage --optimize --metrics alpha --alpha 0.5 --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing vanilla --encoder h2gcn --optimize --metrics alpha --alpha 0.5 --preprocessing undersampling --runs 5 --device 'cuda:4'

python3 main.py --dataset pokec_z_large --inprocessing vanilla --encoder gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing vanilla --encoder gat --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing vanilla --encoder sage --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing vanilla --encoder h2gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_z_large --inprocessing fairgnn --encoder gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing fairgnn --encoder gat --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing fairgnn --encoder sage --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing fairgnn --encoder h2gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_z_large --inprocessing nifty --encoder gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing nifty --encoder gat --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing nifty --encoder sage --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing nifty --encoder h2gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_z_large --inprocessing fairsin --encoder gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing fairsin --encoder gat --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing fairsin --encoder sage --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing fairsin --encoder h2gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_z_large --inprocessing fairdrop --encoder gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing fairdrop --encoder gat --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing fairdrop --encoder sage --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing fairdrop --encoder h2gcn --optimize --metrics f1  --preprocessing None --runs 5 --device 'cuda:4'

python3 main.py --dataset pokec_z_large --inprocessing vanilla --encoder gcn --optimize --metrics f1 --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing vanilla --encoder gat --optimize --metrics f1 --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing vanilla --encoder sage --optimize --metrics f1 --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z_large --inprocessing vanilla --encoder h2gcn --optimize --metrics f1 --preprocessing undersampling --runs 5 --device 'cuda:4'