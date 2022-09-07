import numpy as np
import pandas
import matplotlib.pyplot as plt
from tqdm import tqdm
from pyjet import cluster, DTYPE_PTEPM
import os

_1st_hist_type = "stepfilled"
hist_type = 'step' 

plt.rcParams["figure.figsize"] = (10, 8)
# data_base_dir = "/global/home/users/yifengh3/VAE/vec_data/"
# save_plot_dir = "/global/home/users/yifengh3/VAE/vec_data/weighted_plots/"


def plot_eta(data_files, data_names, save_plot_dir=None, show_plot = True):
    plt.figure()
    for index,name,df in zip(range(len(data_files)),data_names,data_files):
        if index==0:
            n,b,_= plt.hist(df[:,:,1].flatten(),label=name, alpha=0.5, histtype=_1st_hist_type, density=True)
        else:
            plt.hist(df[:,:,1].flatten(), label=name, bins=b, alpha=0.5, histtype=hist_type, density=True)
    plt.xlabel("Constituent $\eta$")
    plt.legend()
    if save_plot_dir:
        plt.savefig(os.path.join(save_plot_dir,"Constituent_eta.png"))
    if show_plot:
        plt.show()
    plt.close()

def plot_phi(data_files, data_names, save_plot_dir=None, show_plot = True):
    plt.figure()
    for index,name,df in zip(range(len(data_files)),data_names,data_files):
        if index==0:
            n,b,_= plt.hist(df[:,:,2].flatten(),label=name, alpha=0.5, histtype=_1st_hist_type, density=True)
        else:
            plt.hist(df[:,:,2].flatten(), label=name, bins=b, alpha=0.5, histtype=hist_type, density=True)
    plt.xlabel("Constituent $\phi$")
    plt.legend()
    if save_plot_dir:
        plt.savefig(os.path.join(save_plot_dir,"Constituent_phi.png"))
    if show_plot:
        plt.show()
    plt.close()


def plot_pt_frac(data_files, data_names, save_plot_dir=None, show_plot = True):
    plt.figure()
    for index,name,df in zip(range(len(data_files)),data_names,data_files):
        if index==0:
            n,b,_= plt.hist(df[:,:,0].flatten(),label=name, alpha=0.5, histtype=_1st_hist_type, density=True)
        else:
            plt.hist(df[:,:,0].flatten(), label=name, bins=b, alpha=0.5, histtype=hist_type, density=True)
    plt.xlabel("Constituent $p_T$ fraction")
    plt.yscale("log")
    plt.legend()
    if save_plot_dir:
        plt.savefig(os.path.join(save_plot_dir,"Constituent_pt.png"))
    if show_plot:
        plt.show()
    plt.close()



def event_mass(myinput, pbar=True):
    ms = []
    for i in tqdm(range(len(myinput)), disable= not pbar):
        px = np.sum(myinput[i,:,0].flatten()*np.cos(myinput[i,:,2].flatten()))
        py = np.sum(myinput[i,:,0].flatten()*np.sin(myinput[i,:,2].flatten()))
        pz = np.sum(myinput[i,:,0].flatten()*np.sinh(myinput[i,:,1].flatten()))
        E = np.sum(myinput[i,:,0].flatten()*np.cosh(myinput[i,:,1].flatten()))
        ms += [(E**2-px*px-py*py-pz*pz)**0.5]
    return np.array(ms)


def plot_em(data_files, data_names, save_plot_dir=None, show_plot = True, pbar=True):
    plt.figure()
    for index,name,df in zip(range(len(data_files)),data_names,data_files):
        if index==0:
            n,b,_= plt.hist(event_mass(df, pbar),label=name, alpha=0.5, histtype=_1st_hist_type, density=True)
        else:
            plt.hist(event_mass(df, pbar), label=name, bins=b, alpha=0.5, histtype=hist_type, density=True)
    plt.xlabel("Event Mass")
    plt.legend()
    if save_plot_dir:
        plt.savefig(os.path.join(save_plot_dir,"event_mass.png"))
    if show_plot:
        plt.show()
    plt.close()


def MET(myinput, pbar=True):
    ms = []
    for i in tqdm(range(len(myinput)), disable= not pbar):
        px = np.sum(myinput[i,:,0].flatten()*np.cos(myinput[i,:,2].flatten()))
        py = np.sum(myinput[i,:,0].flatten()*np.sin(myinput[i,:,2].flatten()))
        ms += [np.sqrt(px**2+py**2)]
    return np.array(ms)


def plot_missing_m(data_files, data_names, save_plot_dir=None, show_plot = True, pbar = True):
    plt.figure()
    for index,name,df in zip(range(len(data_files)),data_names,data_files):
        if index==0:
            n,b,_= plt.hist(MET(df, pbar),label=name, alpha=0.5, histtype=_1st_hist_type, density=True)
        else:
            plt.hist(MET(df, pbar), label=name, bins=b, alpha=0.5, histtype=hist_type, density=True)
    plt.xlabel("Missing Momentum")
    plt.legend()
    plt.yscale("log")
    if save_plot_dir:
        plt.savefig(os.path.join(save_plot_dir,"missing_momentum.png"))
    if show_plot:
        plt.show()
    plt.close()




def jet_clustering(ojs, ptmin, pbar = True):
#     print("clustering jets with paramert ptmin={}".format(ptmin))
    njets = []
    pTleadjet = []
    mleadjet = []
    for k in tqdm(range(len(ojs)), disable= not pbar):
        pseudojets_input = np.zeros(50, dtype=DTYPE_PTEPM)
        for i in range(50):
            pseudojets_input[i]['pT'] = ojs[k, i, 0]
            pseudojets_input[i]['eta'] = ojs[k, i, 1]
            pseudojets_input[i]['phi'] = ojs[k, i, 2]
        sequence = cluster(pseudojets_input, R=0.4, p=-1)
        jets = sequence.inclusive_jets(ptmin=ptmin)  # 5 gev
        njets += [len(jets)]
        if (len(jets) > 0):
            pTleadjet += [jets[0].pt]
            mleadjet += [jets[0].mass]
    return njets, pTleadjet, mleadjet



def plot_clustering(data_files, data_names, save_plot_dir=None, show_plot = True, pbar=True):
    plot_label = ["n jets", "pT lead jet", "m lead jet"]
    _ptmin = 10
    _bins = None
    clustering_list = []
    for df in data_files:
        clustering_list.append(jet_clustering(df,_ptmin, pbar))
    
    for i in range(len(plot_label)):
        plt.figure()
        for index,name,df in zip(range(len(data_files)),data_names,clustering_list):
            if index==0 and i!=0:
                n,b,_= plt.hist(df[i],label=name, alpha=0.5, histtype=_1st_hist_type, density=True, bins =np.logspace(1,3.5,20))
            elif index==0 and i==0:
                n,b,_= plt.hist(df[i],label=name, alpha=0.5, histtype=_1st_hist_type, density=True, bins =20, range=(0,20))
            else:
                plt.hist(df[i], label=name, bins=b, alpha=0.5, histtype=hist_type, density=True)
        plt.xlabel(plot_label[i])
        if i!=0:
            plt.semilogx()
        plt.legend()
        if i!=0:
            plt.yscale("log")
        if save_plot_dir:
            plt.savefig(os.path.join(save_plot_dir,plot_label[i].replace(" ","_")+".png"))
        if show_plot:
            plt.show()
        plt.close()



    

def plot_everything(data_files, data_names, save_plot_dir=None, show_plot = True, pbar = True):
    plot_eta(data_files, data_names, save_plot_dir, show_plot)
    plot_phi(data_files, data_names, save_plot_dir, show_plot)
    plot_pt_frac(data_files, data_names, save_plot_dir, show_plot)
    plot_em(data_files, data_names, save_plot_dir, show_plot, pbar)
    plot_missing_m(data_files, data_names, save_plot_dir, show_plot, pbar)
    plot_clustering(data_files, data_names, save_plot_dir, show_plot, pbar)
    
    
def plot_recon_jet(data, log_betas, plot_root_dir):
    for i,jet in enumerate(tqdm(data)):
        save_dir = os.path.join(plot_root_dir,"recon_plot_logbeta={:.2f}".format(log_betas[i]))
        try: 
            os.mkdir(save_dir) 
        except:
            pass
        plot_everything([jet,], ["recon_plot_logbeta={:.2f}".format(log_betas[i]),], save_plot_dir = save_dir, show_plot=False, pbar=False)
        

def compare_recons(original_data,data, log_betas, plot_root_dir, sub_dir_name="comparison", split=5, show_plot=True):
    step = data.shape[0]//split
    split_index = list(range(0,data.shape[0], step))
    sub_sample = data[split_index]
    sub_betas = log_betas[split_index]
    
    plot_entries = [original_data, *sub_sample]
    plot_name = ["original", *["recon_plot_logbeta={:.2f}".format(beta) for beta in sub_betas]]
    save_dir = os.path.join(plot_root_dir,sub_dir_name)
    try: 
        os.mkdir(save_dir) 
    except:
        pass
    
    plot_everything(plot_entries, plot_name , save_plot_dir = save_dir, show_plot=show_plot, pbar=False)