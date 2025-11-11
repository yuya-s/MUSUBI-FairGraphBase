def flip_sens(x, sens_idx):  #全ノードのセンシティブ特徴を反転. niftyと同様 

    x = x.clone()
    sens_mask = (x[:, sens_idx] == 0) | (x[:, sens_idx] == 1)
    x[:, sens_idx][sens_mask] = 1 - x[:, sens_idx][sens_mask]
    
    return x
