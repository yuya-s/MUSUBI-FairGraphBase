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
##
##
### dataset=pokec_n, preprocessing=undersampling, metrics=alpha, alpha=0.5
python3 main.py --dataset pokec_n --inprocessing vanilla --encoder gcn --optimize --metrics alpha --alpha 0.5 --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing vanilla --encoder gat --optimize --metrics alpha --alpha 0.5 --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing vanilla --encoder sage --optimize --metrics alpha --alpha 0.5 --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing vanilla --encoder h2gcn --optimize --metrics alpha --alpha 0.5 --preprocessing undersampling --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_n --inprocessing fairgnn --encoder gcn --optimize --metrics alpha --alpha 0.5 --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing fairgnn --encoder gat --optimize --metrics alpha --alpha 0.5 --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing fairgnn --encoder sage --optimize --metrics alpha --alpha 0.5 --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing fairgnn --encoder h2gcn --optimize --metrics alpha --alpha 0.5 --preprocessing undersampling --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_n --inprocessing nifty --encoder gcn --optimize --metrics alpha --alpha 0.5 --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing nifty --encoder gat --optimize --metrics alpha --alpha 0.5 --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing nifty --encoder sage --optimize --metrics alpha --alpha 0.5 --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing nifty --encoder h2gcn --optimize --metrics alpha --alpha 0.5 --preprocessing undersampling --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_n --inprocessing fairsin --encoder gcn --optimize --metrics alpha --alpha 0.5 --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing fairsin --encoder gat --optimize --metrics alpha --alpha 0.5 --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing fairsin --encoder sage --optimize --metrics alpha --alpha 0.5 --preprocessing undersampling --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing fairsin --encoder h2gcn --optimize --metrics alpha --alpha 0.5 --preprocessing undersampling --runs 5 --device 'cuda:4'
##
##
##
### dataset=pokec_n, preprocessing=bind, metrics=f1
python3 main.py --dataset pokec_n --inprocessing vanilla --encoder gcn --optimize --metrics f1 --bind_enable --bind_del_rate 10 --preprocessing bind --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing vanilla --encoder gat --optimize --metrics f1 --bind_enable --bind_del_rate 10 --preprocessing bind --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing vanilla --encoder sage --optimize --metrics f1 --bind_enable --bind_del_rate 10 --preprocessing bind --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing vanilla --encoder h2gcn --optimize --metrics f1 --bind_enable --bind_del_rate 10 --preprocessing bind --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_n --inprocessing fairgnn --encoder gcn --optimize --metrics f1 --bind_enable --bind_del_rate 10 --preprocessing bind --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing fairgnn --encoder gat --optimize --metrics f1 --bind_enable --bind_del_rate 10 --preprocessing bind --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing fairgnn --encoder sage --optimize --metrics f1 --bind_enable --bind_del_rate 10 --preprocessing bind --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing fairgnn --encoder h2gcn --optimize --metrics f1 --bind_enable --bind_del_rate 10 --preprocessing bind --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_n --inprocessing nifty --encoder gcn --optimize --metrics f1 --bind_enable --bind_del_rate 10 --preprocessing bind --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing nifty --encoder gat --optimize --metrics f1 --bind_enable --bind_del_rate 10 --preprocessing bind --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing nifty --encoder sage --optimize --metrics f1 --bind_enable --bind_del_rate 10 --preprocessing bind --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing nifty --encoder h2gcn --optimize --metrics f1 --bind_enable --bind_del_rate 10 --preprocessing bind --runs 5 --device 'cuda:4'
##
python3 main.py --dataset pokec_n --inprocessing fairsin --encoder gcn --optimize --metrics f1 --bind_enable --bind_del_rate 10 --preprocessing bind --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing fairsin --encoder gat --optimize --metrics f1 --bind_enable --bind_del_rate 10 --preprocessing bind --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing fairsin --encoder sage --optimize --metrics f1 --bind_enable --bind_del_rate 10 --preprocessing bind --runs 5 --device 'cuda:4'
python3 main.py --dataset pokec_n --inprocessing fairsin --encoder h2gcn --optimize --metrics f1 --bind_enable --bind_del_rate 10 --preprocessing bind --runs 5 --device 'cuda:4'
