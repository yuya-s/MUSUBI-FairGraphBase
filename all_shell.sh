### sample shell ###
###
###
### dataset=pokec_n, preprocessing=None, metrics=acc
python3 main.py --dataset pokec_n --inprocessing vanilla --encoder gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing vanilla --encoder gat --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing vanilla --encoder sage --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing vanilla --encoder h2gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_n --inprocessing fairgnn --encoder gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing fairgnn --encoder gat --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing fairgnn --encoder sage --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing fairgnn --encoder h2gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_n --inprocessing nifty --encoder gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing nifty --encoder gat --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing nifty --encoder sage --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing nifty --encoder h2gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_n --inprocessing fairsin --encoder gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing fairsin --encoder gat --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing fairsin --encoder sage --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing fairsin --encoder h2gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_n --inprocessing fairdrop --encoder gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing fairdrop --encoder gat --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing fairdrop --encoder sage --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing fairdrop --encoder h2gcn --optimize --metrics acc  --preprocessing None --runs 5 --device 'cuda:4'

python3 main.py --dataset pokec_n --inprocessing vanilla --encoder gcn --optimize --metrics acc --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing vanilla --encoder gat --optimize --metrics acc --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing vanilla --encoder sage --optimize --metrics acc --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing vanilla --encoder h2gcn --optimize --metrics acc --preprocessing undersampling --runs 5 --device 'cuda:4'

python3 main.py --dataset pokec_n --inprocessing vanilla --encoder gcn --optimize --metrics acc --preprocessing bind --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing vanilla --encoder gat --optimize --metrics acc --preprocessing bind --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing vanilla --encoder sage --optimize --metrics acc --preprocessing bind --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing vanilla --encoder h2gcn --optimize --metrics acc --preprocessing bind --runs 5 --device 'cuda:4'





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

python3 main.py --dataset pokec_n --inprocessing vanilla --encoder gcn --optimize --metrics alpha --alpha 0.5 --preprocessing bind --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing vanilla --encoder gat --optimize --metrics alpha --alpha 0.5 --preprocessing bind --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing vanilla --encoder sage --optimize --metrics alpha --alpha 0.5 --preprocessing bind --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing vanilla --encoder h2gcn --optimize --metrics alpha --alpha 0.5 --preprocessing bind --runs 5 --device 'cuda:4'


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

python3 main.py --dataset pokec_n --inprocessing vanilla --encoder gcn --optimize --metrics f1 --preprocessing bind --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing vanilla --encoder gat --optimize --metrics f1 --preprocessing bind --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing vanilla --encoder sage --optimize --metrics f1 --preprocessing bind --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing vanilla --encoder h2gcn --optimize --metrics f1 --preprocessing bind --runs 5 --device 'cuda:4'



### sample shell ###
###
###
### dataset=pokec_z, preprocessing=None, metrics=acc
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

python3 main.py --dataset pokec_z --inprocessing vanilla --encoder gcn --optimize --metrics acc --preprocessing bind --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing vanilla --encoder gat --optimize --metrics acc --preprocessing bind --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing vanilla --encoder sage --optimize --metrics acc --preprocessing bind --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing vanilla --encoder h2gcn --optimize --metrics acc --preprocessing bind --runs 5 --device 'cuda:4'





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

python3 main.py --dataset pokec_z --inprocessing vanilla --encoder gcn --optimize --metrics alpha --alpha 0.5 --preprocessing bind --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing vanilla --encoder gat --optimize --metrics alpha --alpha 0.5 --preprocessing bind --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing vanilla --encoder sage --optimize --metrics alpha --alpha 0.5 --preprocessing bind --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing vanilla --encoder h2gcn --optimize --metrics alpha --alpha 0.5 --preprocessing bind --runs 5 --device 'cuda:4'


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

python3 main.py --dataset pokec_z --inprocessing vanilla --encoder gcn --optimize --metrics f1 --preprocessing bind --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing vanilla --encoder gat --optimize --metrics f1 --preprocessing bind --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing vanilla --encoder sage --optimize --metrics f1 --preprocessing bind --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_z --inprocessing vanilla --encoder h2gcn --optimize --metrics f1 --preprocessing bind --runs 5 --device 'cuda:4'
