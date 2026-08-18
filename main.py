import argparse
import csv
import json
import platform
import warnings
import copy
import pickle

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = "42"

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import gc

import numpy as np
import yaml

import optuna
from tqdm import tqdm
from utils.dataset import *
from methods.trainer_fairgnn import run_trial_fairgnn
from methods.trainer_fairsin import run_trial_fairsin
from methods.trainer_nifty import run_trial_nifty
from methods.trainer_vanilla import run_trial_vanilla
from methods.trainer_fairdrop import run_trial_fairdrop
from methods.trainer_fairgb import run_trial_fairgb
from methods.trainer_fairgt import run_trial_fairgt
from methods.trainer_dabgnn import run_trial_dabgnn
from methods.bind import run_bind
from methods.undersampling import run_undersampling
from methods.fairdrop import run_fairdrop
from utils.utils import *
warnings.filterwarnings('ignore')
import time

def get_run_trial_func(inprocessing):

    # inprocessing select
    if inprocessing == 'fairsin':
        return run_trial_fairsin
    elif inprocessing == 'vanilla':
        return run_trial_vanilla
    elif inprocessing == 'fairgnn':
        return run_trial_fairgnn
    elif inprocessing == 'nifty':
        return run_trial_nifty
    elif inprocessing == 'fairdrop':
        return run_trial_fairdrop
    elif inprocessing == 'fairgb':
        return run_trial_fairgb
    elif inprocessing == 'fairgt':
        return run_trial_fairgt
    elif inprocessing == 'dabgnn':
        return run_trial_dabgnn

def get_run_preprocessing_func(preprocessing):

    # preprocessing select
    if preprocessing == 'undersampling':
        return run_undersampling
    elif preprocessing == 'bind':
        return run_bind
    elif preprocessing == 'fairdrop':
        return run_fairdrop

def get_inprocessing_param(params, inprocessing, args, param_key):
    if args.dataset == 'wikidata':
        return params[inprocessing][args.encoder][param_key]
    return params[inprocessing][param_key]

def run(data, args):
    # objective start
    def objective(trial):

        # inprocessing select
        #print(f"[DEBUG] Trial {trial.number} start, GPU mem allocated:",torch.cuda.memory_allocated()/1024**3, "GB","reserved:", torch.cuda.memory_reserved()/1024**3, "GB",flush=True)
        try:
            run_trial = get_run_trial_func(args.inprocessing)

            if args.inprocessing == 'vanilla':
                # optimize param
                vanilla_optimize_param = get_inprocessing_param(params, "vanilla", args, "optimize_param")
                vanilla_optimize_hidden = vanilla_optimize_param["hidden"]
                vanilla_optimize_gnn_layer_size = vanilla_optimize_param["gnn_layer_size"]
                vanilla_optimize_gnn_hidden = vanilla_optimize_param["gnn_hidden"]
                vanilla_optimize_cls_layer_size = vanilla_optimize_param["cls_layer_size"]
                vanilla_optimize_lr = vanilla_optimize_param["lr"]
                vanilla_optimize_weight_decay = vanilla_optimize_param["weight_decay"]

                args.hidden = trial.suggest_categorical("hidden", vanilla_optimize_hidden)
                args.gnn_layer_size = trial.suggest_categorical("gnn_layer_size", vanilla_optimize_gnn_layer_size)
                args.gnn_hidden = trial.suggest_categorical("gnn_hidden", vanilla_optimize_gnn_hidden)
                args.cls_layer_size = trial.suggest_categorical("cls_layer_size", vanilla_optimize_cls_layer_size)
                args.lr = trial.suggest_categorical("lr", vanilla_optimize_lr)
                args.weight_decay = trial.suggest_categorical("weight_decay", vanilla_optimize_weight_decay)

            elif args.inprocessing == 'fairgnn':
                # optimize param
                fairgnn_optimize_param = get_inprocessing_param(params, "fairgnn", args, "optimize_param")
                fairgnn_optimize_hidden = fairgnn_optimize_param["hidden"]
                fairgnn_optimize_gnn_layer_size = fairgnn_optimize_param["gnn_layer_size"]
                fairgnn_optimize_gnn_hidden = fairgnn_optimize_param["gnn_hidden"]
                fairgnn_optimize_cls_layer_size = fairgnn_optimize_param["cls_layer_size"]
                fairgnn_optimize_acc = fairgnn_optimize_param["acc"]
                fairgnn_optimize_g_alpha = fairgnn_optimize_param["g_alpha"]
                fairgnn_optimize_g_beta = fairgnn_optimize_param["g_beta"]
                fairgnn_optimize_proj_hidden = fairgnn_optimize_param["proj_hidden"]
                fairgnn_optimize_lr = fairgnn_optimize_param["lr"]
                fairgnn_optimize_weight_decay = fairgnn_optimize_param["weight_decay"]

                args.hidden = trial.suggest_categorical("hidden", fairgnn_optimize_hidden)
                args.gnn_layer_size = trial.suggest_categorical("gnn_layer_size", fairgnn_optimize_gnn_layer_size)
                args.gnn_hidden = trial.suggest_categorical("gnn_hidden", fairgnn_optimize_gnn_hidden)
                args.cls_layer_size = trial.suggest_categorical("cls_layer_size", fairgnn_optimize_cls_layer_size)
                args.acc = trial.suggest_categorical("acc", fairgnn_optimize_acc)
                args.g_alpha = trial.suggest_categorical("g_alpha", fairgnn_optimize_g_alpha)
                args.g_beta = trial.suggest_categorical("g_beta", fairgnn_optimize_g_beta)
                args.proj_hidden = trial.suggest_categorical("proj_hidden", fairgnn_optimize_proj_hidden)
                args.lr = trial.suggest_categorical("lr", fairgnn_optimize_lr)
                args.weight_decay = trial.suggest_categorical("weight_decay", fairgnn_optimize_weight_decay)

            elif args.inprocessing == 'nifty':
                # optimize param
                nifty_optimize_param = get_inprocessing_param(params, "nifty", args, "optimize_param")
                nifty_optimize_hidden = nifty_optimize_param["hidden"]
                nifty_optimize_gnn_layer_size = nifty_optimize_param["gnn_layer_size"]
                nifty_optimize_gnn_hidden = nifty_optimize_param["gnn_hidden"]
                nifty_optimize_cls_layer_size = nifty_optimize_param["cls_layer_size"]
                nifty_optimize_num_proj_hidden = nifty_optimize_param["num_proj_hidden"]
                nifty_optimize_lr = nifty_optimize_param["lr"]
                nifty_optimize_weight_decay = nifty_optimize_param["weight_decay"]
                nifty_optimize_sim_coeff = nifty_optimize_param["sim_coeff"]
                nifty_optimize_drop_edge_rate_1 = nifty_optimize_param["drop_edge_rate_1"]
                nifty_optimize_drop_edge_rate_2 = nifty_optimize_param["drop_edge_rate_2"]
                nifty_optimize_drop_feature_rate_1 = nifty_optimize_param["drop_feature_rate_1"]
                nifty_optimize_drop_feature_rate_2 = nifty_optimize_param["drop_feature_rate_2"]

                args.hidden = trial.suggest_categorical("hidden", nifty_optimize_hidden)
                args.gnn_layer_size = trial.suggest_categorical("gnn_layer_size", nifty_optimize_gnn_layer_size)
                args.gnn_hidden = trial.suggest_categorical("gnn_hidden", nifty_optimize_gnn_hidden)
                args.cls_layer_size = trial.suggest_categorical("cls_layer_size", nifty_optimize_cls_layer_size)
                args.num_proj_hidden = trial.suggest_categorical("num_proj_hidden", nifty_optimize_num_proj_hidden)
                args.lr = trial.suggest_categorical("lr", nifty_optimize_lr)
                args.weight_decay = trial.suggest_categorical("weight_decay", nifty_optimize_weight_decay)
                args.sim_coeff = trial.suggest_categorical("sim_coeff", nifty_optimize_sim_coeff)
                args.drop_edge_rate_1 = trial.suggest_categorical("drop_edge_rate_1", nifty_optimize_drop_edge_rate_1)
                args.drop_edge_rate_2 = trial.suggest_categorical("drop_edge_rate_2", nifty_optimize_drop_edge_rate_2)
                args.drop_feature_rate_1 = trial.suggest_categorical("drop_feature_rate_1", nifty_optimize_drop_feature_rate_1)
                args.drop_feature_rate_2 = trial.suggest_categorical("drop_feature_rate_2", nifty_optimize_drop_feature_rate_2)
            elif args.inprocessing == 'fairsin':
                # optimize param
                fairsin_optimize_param = get_inprocessing_param(params, "fairsin", args, "optimize_param")
                fairsin_optimize_hidden = fairsin_optimize_param["hidden"]
                fairsin_optimize_gnn_layer_size = fairsin_optimize_param["gnn_layer_size"]
                fairsin_optimize_gnn_hidden = fairsin_optimize_param["gnn_hidden"]
                fairsin_optimize_cls_layer_size = fairsin_optimize_param["cls_layer_size"]
                fairsin_optimize_c_lr = fairsin_optimize_param["c_lr"]
                fairsin_optimize_e_lr = fairsin_optimize_param["e_lr"]
                fairsin_optimize_m_lr = fairsin_optimize_param["m_lr"]
                fairsin_optimize_d_lr = fairsin_optimize_param["d_lr"]
                fairsin_optimize_delta = fairsin_optimize_param["delta"]
                fairsin_optimize_wd = fairsin_optimize_param["wd"]
                #fairsin_optimize_dropout = fairsin_optimize_param["dropout"]
                # fixed param
                fairsin_fixed_param = get_inprocessing_param(params, "fairsin", args, "fixed_param")
                fairsin_fixed_param_epochs = fairsin_fixed_param["epochs"]
                fairsin_fixed_param_d_epochs = fairsin_fixed_param["d_epochs"]
                fairsin_fixed_param_c_epochs = fairsin_fixed_param["c_epochs"]
                fairsin_fixed_param_m_epochs = fairsin_fixed_param["m_epochs"]
                fairsin_fixed_param_d = fairsin_fixed_param["d"]

                args.hidden = trial.suggest_categorical('hidden', fairsin_optimize_hidden)
                args.gnn_layer_size = trial.suggest_categorical('gnn_layer_size', fairsin_optimize_gnn_layer_size)
                args.gnn_hidden = trial.suggest_categorical('gnn_hidden', fairsin_optimize_gnn_hidden)
                args.cls_layer_size = trial.suggest_categorical('cls_layer_size', fairsin_optimize_cls_layer_size)
                args.c_lr = trial.suggest_categorical('c_lr', fairsin_optimize_c_lr)
                args.e_lr = trial.suggest_categorical('e_lr', fairsin_optimize_e_lr)
                args.m_lr = trial.suggest_categorical('m_lr', fairsin_optimize_m_lr)
                args.d_lr = trial.suggest_categorical('d_lr', fairsin_optimize_d_lr)
                args.delta = trial.suggest_categorical('delta', fairsin_optimize_delta)
                wd = trial.suggest_categorical('wd', fairsin_optimize_wd)
                #args.dropout = trial.suggest_categorical('dropout', fairsin_optimize_dropout)
                args.c_wd = wd
                args.d_wd = wd
                args.e_wd = wd

                args.epochs = fairsin_fixed_param_epochs
                args.d_epochs = fairsin_fixed_param_d_epochs
                args.c_epochs = fairsin_fixed_param_c_epochs
                args.m_epoch = fairsin_fixed_param_m_epochs
                args.d = fairsin_fixed_param_d

            elif args.inprocessing == 'fairdrop':
                # optimize param
                fairdrop_optimize_hidden = params["fairdrop"]["optimize_param"]["hidden"]
                fairdrop_optimize_gnn_layer_size = params["fairdrop"]["optimize_param"]["gnn_layer_size"]
                fairdrop_optimize_gnn_hidden = params["fairdrop"]["optimize_param"]["gnn_hidden"]
                fairdrop_optimize_cls_layer_size = params["fairdrop"]["optimize_param"]["cls_layer_size"]
                fairdrop_optimize_lr = params["fairdrop"]["optimize_param"]["lr"]
                fairdrop_optimize_weight_decay = params["fairdrop"]["optimize_param"]["weight_decay"]
                fairdrop_optimize_delta = params["fairdrop"]["optimize_param"]["delta"]

                args.hidden = trial.suggest_categorical("hidden", fairdrop_optimize_hidden)
                args.gnn_layer_size = trial.suggest_categorical("gnn_layer_size", fairdrop_optimize_gnn_layer_size)
                args.gnn_hidden = trial.suggest_categorical("gnn_hidden", fairdrop_optimize_gnn_hidden)
                args.cls_layer_size = trial.suggest_categorical("cls_layer_size", fairdrop_optimize_cls_layer_size)
                args.lr = trial.suggest_categorical("lr", fairdrop_optimize_lr)
                args.weight_decay = trial.suggest_categorical("weight_decay", fairdrop_optimize_weight_decay)
                args.delta = trial.suggest_categorical("delta", fairdrop_optimize_delta)

            elif args.inprocessing == 'fairgb':
                # optimize param
                fairgb_optimize_hidden = params["fairgb"]["optimize_param"]["hidden"]
                fairgb_optimize_gnn_layer_size = params["fairgb"]["optimize_param"]["gnn_layer_size"]
                fairgb_optimize_gnn_hidden = params["fairgb"]["optimize_param"]["gnn_hidden"]
                fairgb_optimize_cls_layer_size = params["fairgb"]["optimize_param"]["cls_layer_size"]
                fairgb_optimize_e_lr = params["fairgb"]["optimize_param"]["e_lr"]
                fairgb_optimize_c_lr = params["fairgb"]["optimize_param"]["c_lr"]
                fairgb_optimize_wd = params["fairgb"]["optimize_param"]["wd"]
                fairgb_optimize_eta = params["fairgb"]["optimize_param"]["eta"]
                fairgb_optimize_warmup = params["fairgb"]["optimize_param"]["warmup"]

                args.hidden = trial.suggest_categorical("hidden", fairgb_optimize_hidden)
                args.gnn_layer_size = trial.suggest_categorical("gnn_layer_size", fairgb_optimize_gnn_layer_size)
                args.gnn_hidden = trial.suggest_categorical("gnn_hidden", fairgb_optimize_gnn_hidden)
                args.cls_layer_size = trial.suggest_categorical("cls_layer_size", fairgb_optimize_cls_layer_size)
                args.e_lr = trial.suggest_categorical("e_lr", fairgb_optimize_e_lr)
                args.c_lr = trial.suggest_categorical("c_lr", fairgb_optimize_c_lr)
                args.wd = trial.suggest_categorical("wd", fairgb_optimize_wd)
                args.eta = trial.suggest_categorical("eta", fairgb_optimize_eta)
                args.warmup = trial.suggest_categorical("warmup", fairgb_optimize_warmup)
            
            elif args.inprocessing == 'fairgt':
                # optimize param
                fairgt_optimize_hidden = params["fairgt"]["optimize_param"]["hidden"]
                fairgt_optimize_n_layers = params["fairgt"]["optimize_param"]["n_layers"]
                fairgt_optimize_n_heads = params["fairgt"]["optimize_param"]["n_heads"]                
                fairgt_optimize_pe_dim = params["fairgt"]["optimize_param"]["pe_dim"]     
                fairgt_optimize_cliques = params["fairgt"]["optimize_param"]["cliques"]  
                fairgt_optimize_hops = params["fairgt"]["optimize_param"]["hops"]                  
                fairgt_optimize_cls_layer_size = params["fairgt"]["optimize_param"]["cls_layer_size"]
                fairgt_optimize_lr = params["fairgt"]["optimize_param"]["lr"]
                fairgt_optimize_weight_decay = params["fairgt"]["optimize_param"]["weight_decay"]
                fairgt_optimize_dropout = params["fairgt"]["optimize_param"]["dropout"]

                args.hidden = trial.suggest_categorical("hidden", fairgt_optimize_hidden)
                args.n_layers = trial.suggest_categorical("n_layers", fairgt_optimize_n_layers)
                args.n_heads = trial.suggest_categorical("n_heads", fairgt_optimize_n_heads)
                args.pe_dim = trial.suggest_categorical("pe_dim", fairgt_optimize_pe_dim)
                args.cliques = trial.suggest_categorical("cliques", fairgt_optimize_cliques)
                args.hops = trial.suggest_categorical("hops", fairgt_optimize_hops)
                args.cls_layer_size = trial.suggest_categorical("cls_layer_size", fairgt_optimize_cls_layer_size)
                args.lr = trial.suggest_categorical("lr", fairgt_optimize_lr)
                args.weight_decay = trial.suggest_categorical("weight_decay", fairgt_optimize_weight_decay)
                args.dropout = trial.suggest_categorical('dropout', fairgt_optimize_dropout)        
            
            elif args.inprocessing == 'dabgnn':
                # optimize param
                dabgnn_optimize_hidden = params["dabgnn"]["optimize_param"]["hidden"]
                dabgnn_optimize_gnn_layer_size = params["dabgnn"]["optimize_param"]["gnn_layer_size"]
                dabgnn_optimize_gnn_hidden = params["dabgnn"]["optimize_param"]["gnn_hidden"]
                dabgnn_optimize_cls_layer_size = params["dabgnn"]["optimize_param"]["cls_layer_size"]
                dabgnn_optimize_s_lr = params["dabgnn"]["optimize_param"]["s_lr"]
                dabgnn_optimize_a_lr = params["dabgnn"]["optimize_param"]["a_lr"]
                dabgnn_optimize_l_lr = params["dabgnn"]["optimize_param"]["l_lr"]
                dabgnn_optimize_c_lr = params["dabgnn"]["optimize_param"]["c_lr"]
                dabgnn_optimize_w_lr = params["dabgnn"]["optimize_param"]["w_lr"]
                dabgnn_optimize_weight_decay = params["dabgnn"]["optimize_param"]["weight_decay"]
                dabgnn_optimize_dis = params["dabgnn"]["optimize_param"]["dis"]
                dabgnn_optimize_l_dis = params["dabgnn"]["optimize_param"]["l_dis"]
                dabgnn_optimize_lambda_gp = params["dabgnn"]["optimize_param"]["lambda_gp"]
                dabgnn_optimize_s_alpha = params["dabgnn"]["optimize_param"]["s_alpha"]
                dabgnn_optimize_a_alpha = params["dabgnn"]["optimize_param"]["a_alpha"]
                dabgnn_optimize_l_alpha = params["dabgnn"]["optimize_param"]["l_alpha"]
                dabgnn_optimize_dis = params["dabgnn"]["optimize_param"]["dis"]
                dabgnn_optimize_l_dis = params["dabgnn"]["optimize_param"]["l_dis"]
                dabgnn_optimize_k = params["dabgnn"]["optimize_param"]["k"]

                args.hidden = trial.suggest_categorical("hidden", dabgnn_optimize_hidden)
                args.gnn_layer_size = trial.suggest_categorical("gnn_layer_size", dabgnn_optimize_gnn_layer_size)
                args.gnn_hidden = trial.suggest_categorical("gnn_hidden", dabgnn_optimize_gnn_hidden)
                args.cls_layer_size = trial.suggest_categorical("cls_layer_size", dabgnn_optimize_cls_layer_size)
                args.s_lr = trial.suggest_categorical("s_lr", dabgnn_optimize_s_lr)
                args.a_lr = trial.suggest_categorical("a_lr", dabgnn_optimize_a_lr)
                args.l_lr = trial.suggest_categorical("l_lr", dabgnn_optimize_l_lr)
                args.c_lr = trial.suggest_categorical("c_lr", dabgnn_optimize_c_lr)
                args.w_lr = trial.suggest_categorical("w_lr", dabgnn_optimize_w_lr)
                args.weight_decay = trial.suggest_categorical("weight_decay", dabgnn_optimize_weight_decay)
                args.dis = trial.suggest_categorical("dis", dabgnn_optimize_dis)
                args.l_dis = trial.suggest_categorical("l_dis", dabgnn_optimize_l_dis)
                args.lambda_gp = trial.suggest_categorical("lambda_gp", dabgnn_optimize_lambda_gp)
                args.s_alpha = trial.suggest_categorical("s_alpha", dabgnn_optimize_s_alpha)
                args.a_alpha = trial.suggest_categorical("a_alpha", dabgnn_optimize_a_alpha)
                args.l_alpha = trial.suggest_categorical("l_alpha", dabgnn_optimize_l_alpha)
                args.k = trial.suggest_categorical("k", dabgnn_optimize_k)
            t_total = time.time()
            all_metrics, best_eval_output =\
                run_trial(data, args, args.trial_count)
            train_time_all.append(time.time() - t_total)

            
            if args.metrics == "acc":
                val_acc = all_metrics[0].acc
                del all_metrics, best_eval_output
                gc.collect()
                torch.cuda.empty_cache()
                return val_acc
            elif args.metrics == "f1":
                val_f1 = all_metrics[0].f1
                del all_metrics, best_eval_output
                gc.collect()
                torch.cuda.empty_cache()
                return val_f1
            elif args.metrics =="alpha":
                composite = all_metrics[0].acc + all_metrics[0].f1 -  args.alpha * (all_metrics[0].parity + all_metrics[0].equality)
                del all_metrics, best_eval_output
                gc.collect()
                torch.cuda.empty_cache()
                return composite

        except torch.cuda.OutOfMemoryError:
            gc.collect()
            torch.cuda.empty_cache()
            print("OutOfMemory")
            return float(0)

        finally:
            gc.collect()
            torch.cuda.empty_cache()

       # return composite
       # return val_acc
    # objective end

    trial_count = tqdm(range(args.runs), unit='run')

    acc, f1, auc_roc, parity, equality = np.zeros(args.runs), np.zeros(
        args.runs), np.zeros(args.runs), np.zeros(args.runs), np.zeros(args.runs)

    dataoriginal = copy.deepcopy(data)
    #data = data.to(args.device)

    train_time_all = []
    main_total = time.time()

    all_trial_metrics = []


    # trial start
    for tr in trial_count:
        print("starting trial {}".format(tr))

        args.trial_count = tr + 1
        run_preprocessing = get_run_preprocessing_func(args.preprocessing)
        if(args.preprocessing != 'None'):
            data=copy.deepcopy(dataoriginal)
            print("starting preprocessing")
            print(len(data.x))
            print(len(dataoriginal.x))
#            print("before preprocess. train: "+ str(len(np.where(data.train_mask[args.trial_count - 1])[0])) + "val: " + str(len(np.where(data.val_mask[args.trial_count - 1])[0]))+ "test: " + str(len(np.where(data.test_mask[args.trial_count - 1])[0])))
#            print(np.where(data.test_mask[args.trial_count - 1])[0])
            run_preprocessing(data, args, args.trial_count)
#            print("after preprocess. train: "+ str(len(np.where(data.train_mask[args.trial_count - 1])[0])) + "val: " + str(len(np.where(data.val_mask[args.trial_count - 1])[0]))+ "test: " + str(len(np.where(data.test_mask[args.trial_count - 1])[0])))
#            print(np.where(data.test_mask[args.trial_count - 1])[0])
            print("done preprocessing")

        data = data.to(args.device)

        # load config
        config_file = 'config_inprocessing_wikidata.yml' if args.dataset == 'wikidata' else 'config_inprocessing.yml'
        with open(config_file, 'r') as yml:
            params = yaml.safe_load(yml)



        if args.optimize:
            # optimize
            study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(objective, n_trials=args.optrials, n_jobs=1) # optrials=20

            print('Trial:', tr, 'Number of finished study:', len(study.trials))
            print('Trial:', tr, 'Best param:', study.best_trial.params)
            print('Trial:', tr, 'Best score:', study.best_trial.user_attrs)

            # output to json file
            with open(f"{args.output_dir}/bestparam_{tr}.json", "w") as f:
                json.dump(study.best_trial.params, f, indent=4)

        # load from json file
        with open(f"{args.output_dir}/bestparam_{tr}.json") as f:
            hparams = json.load(f)

        # inprocessing select
        run_trial = get_run_trial_func(args.inprocessing)

        if args.inprocessing == 'vanilla':
            args.hidden = hparams['hidden']
            args.gnn_layer_size = hparams['gnn_layer_size']
            args.gnn_hidden = hparams['gnn_hidden']
            args.cls_layer_size = hparams['cls_layer_size']
            args.lr = hparams['lr']
            args.weight_decay = hparams['weight_decay']

        elif args.inprocessing == 'fairgnn':
            args.hidden = hparams['hidden']
            args.gnn_layer_size = hparams['gnn_layer_size']
            args.gnn_hidden = hparams['gnn_hidden']
            args.cls_layer_size = hparams['cls_layer_size']
            args.acc = hparams['acc']
            args.g_alpha = hparams['g_alpha']
            args.g_beta = hparams['g_beta']
            args.proj_hidden = hparams['proj_hidden']
            args.lr = hparams['lr']
            args.weight_decay = hparams['weight_decay']

        elif args.inprocessing == 'nifty':
            args.hidden = hparams['hidden']
            args.gnn_layer_size = hparams['gnn_layer_size']
            args.gnn_hidden = hparams['gnn_hidden']
            args.cls_layer_size = hparams['cls_layer_size']
            args.num_proj_hidden = hparams['num_proj_hidden']
            args.lr = hparams['lr']
            args.weight_decay = hparams['weight_decay']
            args.sim_coeff = hparams['sim_coeff']
            args.drop_edge_rate_1 = hparams['drop_edge_rate_1']
            args.drop_edge_rate_2 = hparams['drop_edge_rate_2']
            args.drop_feature_rate_1 = hparams['drop_feature_rate_1']
            args.drop_feature_rate_2 = hparams['drop_feature_rate_2']
        elif args.inprocessing == 'fairsin':
            fairsin_fixed_param = get_inprocessing_param(params, "fairsin", args, "fixed_param")
            fairsin_fixed_param_epochs = fairsin_fixed_param["epochs"]
            fairsin_fixed_param_d_epochs = fairsin_fixed_param["d_epochs"]
            fairsin_fixed_param_c_epochs = fairsin_fixed_param["c_epochs"]
            fairsin_fixed_param_m_epochs = fairsin_fixed_param["m_epochs"]
            fairsin_fixed_param_d = fairsin_fixed_param["d"]

            args.hidden = hparams['hidden']
            args.gnn_layer_size = hparams['gnn_layer_size']
            args.gnn_hidden = hparams['gnn_hidden']
            args.cls_layer_size = hparams['cls_layer_size']
            args.c_lr = hparams['c_lr']
            args.e_lr = hparams['e_lr']
            args.m_lr = hparams['m_lr']
            args.d_lr = hparams['d_lr']
            args.delta = hparams['delta']
            wd = hparams['wd']
            args.c_wd = wd
            args.d_wd = wd
            args.e_wd = wd
            #args.dropout = hparams['dropout']
            args.epochs = fairsin_fixed_param_epochs
            args.d_epochs = fairsin_fixed_param_d_epochs
            args.c_epochs = fairsin_fixed_param_c_epochs
            args.m_epochs = fairsin_fixed_param_m_epochs
            args.d = fairsin_fixed_param_d
        
        elif args.inprocessing == 'fairdrop':
            args.hidden = hparams['hidden']
            args.gnn_layer_size = hparams['gnn_layer_size']
            args.gnn_hidden = hparams['gnn_hidden']
            args.cls_layer_size = hparams['cls_layer_size']
            args.lr = hparams['lr']                        
            args.weight_decay = hparams['weight_decay']
            args.delta = hparams['delta']     
                        
        elif args.inprocessing == 'fairgb':
            args.hidden = hparams['hidden']
            args.gnn_layer_size = hparams['gnn_layer_size']
            args.gnn_hidden = hparams['gnn_hidden']
            args.cls_layer_size = hparams['cls_layer_size']
            args.e_lr = hparams['e_lr']   
            args.c_lr = hparams['c_lr']            
            args.wd = hparams['wd']         
            args.eta = hparams['eta']
            args.warmup = hparams['warmup']
             
        elif args.inprocessing == 'fairgt':
            args.hidden = hparams['hidden']
            args.n_layers = hparams['n_layers']
            args.n_heads = hparams['n_heads']
            args.cls_layer_size = hparams['cls_layer_size']
            args.lr = hparams['lr']
            args.weight_decay = hparams['weight_decay']
            args.dropout = hparams['dropout']
            args.hops = hparams['hops']
            args.pe_dim = hparams['pe_dim']
            args.cliques = hparams['cliques']    
        
        elif args.inprocessing == 'dabgnn':
            args.hidden = hparams["hidden"]
            args.hidden = hparams['hidden']
            args.gnn_layer_size = hparams['gnn_layer_size']
            args.gnn_hidden = hparams['gnn_hidden']
            args.cls_layer_size = hparams['cls_layer_size']
            args.s_lr = hparams['s_lr']      
            args.a_lr = hparams['a_lr']  
            args.l_lr = hparams['l_lr']
            args.c_lr = hparams['c_lr']  
            args.w_lr = hparams['w_lr']                     
            args.weight_decay = hparams['weight_decay']
            args.dis = hparams['dis'] 
            args.l_dis = hparams['l_dis'] 
            args.lambda_gp = hparams['lambda_gp'] 
            args.s_alpha = hparams['s_alpha'] 
            args.a_alpha = hparams['a_alpha'] 
            args.l_alpha = hparams['l_alpha'] 
            args.k = hparams['k'] 
        t_total = time.time()


        all_metrics, best_output = run_trial(data, args, args.trial_count)


        train_time_all.append(time.time() - t_total)

        main_total_time = time.time() - main_total

        # GPU Usage
        pf = platform.system()
        gpu_usage = 0
        if pf != 'Windows':
            gpu_usage = float(get_gpu_info()[args.gpu_no]['memory.used'])

        all_trial_metrics.append([all_metrics[1], gpu_usage, np.mean(train_time_all), main_total_time, best_output])

    return all_trial_metrics


def main(args):

    # load preprocessing config
    with open('config_preprocessing.yml', 'r') as yml_b:
        params_p = yaml.safe_load(yml_b)

    args.bind_del_rate = params_p["bind"]["bind_del_rate"]
    args.bind_fastmode = params_p["bind"]["bind_fastmode"]
    args.bind_seed = params_p["bind"]["bind_seed"]
    args.bind_epochs = params_p["bind"]["bind_epochs"]
    args.bind_lr = params_p["bind"]["bind_lr"]
    args.bind_weight_decay = params_p["bind"]["bind_weight_decay"]
    args.bind_hidden = params_p["bind"]["bind_hidden"]
    args.bind_dropout = params_p["bind"]["bind_dropout"]
    args.bind_helpfulness_collection = params_p["bind"]["bind_helpfulness_collection"]

    args.fairdrop_delta = params_p["fairdrop"]["delta"]

    metrics_str = args.metrics
    if args.metrics == 'alpha':
        metrics_str = 'alpha' + str(args.alpha)

    if args.preprocessing == "bind":
        output_dir = f'output/{args.dataset}_{args.inprocessing}_{args.encoder}_{metrics_str}_{args.preprocessing}{args.bind_del_rate}'
    elif args.preprocessing == "fairdrop":
        output_dir = f'output/{args.dataset}_{args.inprocessing}_{args.encoder}_{metrics_str}_{args.preprocessing}{args.fairdrop_delta}'
    elif args.inprocessing == "fairgt":
        output_dir = f'output/{args.dataset}_{args.inprocessing}_{metrics_str}_{args.preprocessing}'
    else:
        output_dir = f'output/{args.dataset}_{args.inprocessing}_{args.encoder}_{metrics_str}_{args.preprocessing}'

    args.output_dir = output_dir

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/train_time/', exist_ok=True)

    args.device = args.device.strip()
    args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    args.gpu_no = 0
    if 'cuda:' in str(args.device):
        args.gpu_no = int(str(args.device).replace('cuda:',''))
    torch.cuda.set_device(args.device)

    print(f"run with {args.device}")

    fix_seed(42)

    data, args.sens_idx, args.x_min, args.x_max = get_dataset(args.dataset, args.runs, args)
    args.num_features, args.num_classes = data.x.shape[1], 2-1 # binary classes are 0,1

    all_trial_metrics = run(data, args)

    # output to csv
    with open(f'{output_dir}/all_output.csv', 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(['Inprocessing', args.inprocessing])
        writer.writerow(['Encoder', args.encoder])
        writer.writerow(['Dataset', args.dataset])
        writer.writerow(['Metrics', args.metrics])
        if args.metrics == "alpha":
            writer.writerow(['Alpha', args.alpha])
        else:
            writer.writerow([''])
        writer.writerow([''])
        writer.writerow(['no', 'ACC', 'AUC', 'F1', 'SP', 'EO', 'EA', 'CF', 'GPU Usage', 'Parameter Num', 'Train time', 'Total time'])
        accs, aucs, f1s, paritys, equalitys, equal_accuracys, counterfactual_fairnesss, acc_sens0s, auc_sens0s, f1_sens0s, acc_sens1s, auc_sens1s, f1_sens1s,\
            gpu_usages, parameter_nums, train_times, total_times = \
            [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        for i, trial_metrics in enumerate(all_trial_metrics):
            metrics = trial_metrics[0]
            gpu_usage = trial_metrics[1]
            train_time = trial_metrics[2]
            total_time = trial_metrics[3]
            writer.writerow(
                [(i+1), metrics.acc, metrics.auc, metrics.f1, metrics.parity, metrics.equality,
                 metrics.equal_accuracy, metrics.counterfactual_fairness, gpu_usage,
                 metrics.model_param_cnt, train_time, total_time])
            accs.append(metrics.acc)
            aucs.append(metrics.auc)
            f1s.append(metrics.f1)
            paritys.append(metrics.parity)
            equalitys.append(metrics.equality)
            equal_accuracys.append(metrics.equal_accuracy)
            counterfactual_fairnesss.append(metrics.counterfactual_fairness)
            acc_sens0s.append(metrics.acc_sens0)
            auc_sens0s.append(metrics.auc_sens0)
            f1_sens0s.append(metrics.f1_sens0)
            acc_sens1s.append(metrics.acc_sens1)
            auc_sens1s.append(metrics.auc_sens1)
            f1_sens1s.append(metrics.f1_sens1)
            gpu_usages.append(gpu_usage)
            parameter_nums.append(metrics.model_param_cnt)
            train_times.append(train_time)
            total_times.append(total_time)

        accs_pct = np.array(accs) * 100
        aucs_pct = np.array(aucs) * 100
        f1s_pct = np.array(f1s) * 100
        paritys_pct = np.array(paritys) * 100
        equalitys_pct = np.array(equalitys) * 100
        equal_accuracys_pct = np.array(equal_accuracys) * 100
        counterfactual_fairnesss_pct = np.array(counterfactual_fairnesss) * 100

        writer.writerow(
            ['mean',
             f"{np.round(np.mean(accs_pct), decimals=2)} +- {np.round(np.var(accs_pct), decimals=2)}",
             f"{np.round(np.mean(aucs_pct), decimals=2)} +- {np.round(np.var(aucs_pct), decimals=2)}",
             f"{np.round(np.mean(f1s_pct), decimals=2)} +- {np.round(np.var(f1s_pct), decimals=2)}",
             f"{np.round(np.mean(paritys_pct), decimals=2)} +- {np.round(np.var(paritys_pct), decimals=2)}",
             f"{np.round(np.mean(equalitys_pct), decimals=2)} +- {np.round(np.var(equalitys_pct), decimals=2)}",
             f"{np.round(np.mean(equal_accuracys_pct), decimals=2)} +- {np.round(np.var(equal_accuracys_pct), decimals=2)}",
             f"{np.round(np.mean(counterfactual_fairnesss_pct), decimals=2)} +- {np.round(np.var(counterfactual_fairnesss_pct), decimals=2)}",
             np.mean(gpu_usages), np.mean(parameter_nums), np.mean(train_times), np.mean(total_times)])

        for i, trial_metrics in enumerate(all_trial_metrics):
            writer.writerow([f'trial {i+1}'])
            writer.writerow(['ConfusionMatrics(All)', 'Prediction 0', 'Prediction 1', 'sum',
                             'ConfusionMatrics(sens0)', 'Prediction 0', 'Prediction 1', 'sum',
                             'ConfusionMatrics(sens1)', 'Prediction 0', 'Prediction 1', 'sum'])
            metrics = trial_metrics[0]
            tn, fp, fn, tp = metrics.cm.flatten()
            tn0, fp0, fn0, tp0 = metrics.cm_sens0.flatten()
            tn1, fp1, fn1, tp1 = metrics.cm_sens1.flatten()
            writer.writerow(['Actual 0', tn, fp, (tn+fp),
                             'Actual 0', tn0, fp0, (tn0+fp0),
                             'Actual 0', tn1, fp1, (tn1+fp1)])
            writer.writerow(['Actual 1', fn, tp, (fn+tp),
                             'Actual 1', fn0, tp0, (fn0+tp0),
                             'Actual 1', fn1, tp1, (fn1+tp1)])
            writer.writerow(['sum', (tn+fn), (fp+tp), (tn+fn+fp+tp),
                             'sum', (tn0+fn0), (fp0+tp0), (tn0+fn0+fp0+tp0),
                             'sum', (tn1+fn1), (fp1+tp1), (tn1+fn1+fp1+tp1)])


if __name__ == '__main__':

    fix_seed(42)

    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='pokec_n')
    parser.add_argument('--optimize', action='store_true')
    parser.add_argument('--optrials', type=int, default=20)
    parser.add_argument('--inprocessing', type=str, default='vanilla') # fairsin/vanilla/fairgnn/nifty/fairdrop
    parser.add_argument('--preprocessing', type=str, default='None') # bind/undersampling/fairdrop/None
    parser.add_argument('--trainsize', type=float, default=0.6)
    parser.add_argument('--valsize', type=float, default=0.2)
    parser.add_argument('--encoder', type=str, default='gcn') # gcn/gat/sage
    parser.add_argument('--metrics', type=str, default='acc') # acc/alpha/f1
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--senserandom', type=bool, default=False)
    parser.add_argument('--device', type=str, default='cuda:0') # cuda or cpu


    args = parser.parse_args()
    if args.epochs == -1:
        if args.inprocessing == 'fairsin':
            args.epochs = 50
        if args.inprocessing == 'dabgnn':
            args.epochs = 500
        else:
            args.epochs = 2000

    main(args)
