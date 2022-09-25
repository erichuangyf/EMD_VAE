import numpy as np
import pandas
import matplotlib.pyplot as plt
from tqdm import tqdm
from pyjet import cluster, DTYPE_PTEPM
import os

_1st_hist_type = "stepfilled"
hist_type = 'step' 
_figsize = (15, 12)

# plt.rcParams["figure.figsize"] = (15, 12)
# data_base_dir = "/global/home/users/yifengh3/VAE/vec_data/"
# save_plot_dir = "/global/home/users/yifengh3/VAE/vec_data/weighted_plots/"


def plot_eta(data_files, data_names, save_plot_dir=None, show_plot = True, dpi=200):
    plt.figure(figsize=_figsize)
    for index,name,df in zip(range(len(data_files)),data_names,data_files):
        if index==0:
            n,b,_= plt.hist(df[:,:,1].flatten(),label=name, alpha=0.5, histtype=_1st_hist_type, density=True)
        else:
            plt.hist(df[:,:,1].flatten(), label=name, bins=b, alpha=0.5, histtype=hist_type, density=True)
    plt.xlabel("Constituent $\eta$")
    plt.legend()
    if save_plot_dir:
        plt.savefig(os.path.join(save_plot_dir,"Constituent_eta.png"), dpi=dpi)
    if show_plot:
        plt.show()
    plt.close()

def plot_phi(data_files, data_names, save_plot_dir=None, show_plot = True, dpi=200):
    plt.figure(figsize=_figsize)
    for index,name,df in zip(range(len(data_files)),data_names,data_files):
        if index==0:
            n,b,_= plt.hist(df[:,:,2].flatten(),label=name, alpha=0.5, histtype=_1st_hist_type, density=True)
        else:
            plt.hist(df[:,:,2].flatten(), label=name, bins=b, alpha=0.5, histtype=hist_type, density=True)
    plt.xlabel("Constituent $\phi$")
    plt.legend()
    if save_plot_dir:
        plt.savefig(os.path.join(save_plot_dir,"Constituent_phi.png"), dpi=dpi)
    if show_plot:
        plt.show()
    plt.close()


def plot_pt_frac(data_files, data_names, save_plot_dir=None, show_plot = True, dpi=200):
    plt.figure(figsize=_figsize)
    for index,name,df in zip(range(len(data_files)),data_names,data_files):
        if index==0:
            n,b,_= plt.hist(df[:,:,0].flatten(),label=name, alpha=0.5, histtype=_1st_hist_type, density=True)
        else:
            plt.hist(df[:,:,0].flatten(), label=name, bins=b, alpha=0.5, histtype=hist_type, density=True)
    plt.xlabel("Constituent $p_T$ fraction")
    plt.yscale("log")
    plt.legend()
    if save_plot_dir:
        plt.savefig(os.path.join(save_plot_dir,"Constituent_pt.png"), dpi=dpi)
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


def plot_em(data_files, data_names, save_plot_dir=None, show_plot = True, pbar=True, dpi=200):
    plt.figure(figsize=_figsize)
    for index,name,df in zip(range(len(data_files)),data_names,data_files):
        if index==0:
            n,b,_= plt.hist(event_mass(df, pbar),label=name, alpha=0.5, histtype=_1st_hist_type, density=True)
        else:
            plt.hist(event_mass(df, pbar), label=name, bins=b, alpha=0.5, histtype=hist_type, density=True)
    plt.xlabel("Event Mass")
    plt.legend()
    if save_plot_dir:
        plt.savefig(os.path.join(save_plot_dir,"event_mass.png"), dpi=dpi)
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


def plot_missing_m(data_files, data_names, save_plot_dir=None, show_plot = True, pbar = True, dpi=200):
    b = 100
    plt.figure(figsize=_figsize)
    for index,name,df in zip(range(len(data_files)),data_names,data_files):
        met = MET(df, pbar)
        if index==0:
            n,b,_= plt.hist(met,label=name, alpha=0.5, histtype=_1st_hist_type, density=True, bins=b)
        else:
            plt.hist(met, label=name, bins=b, alpha=0.5, histtype=hist_type, density=True)
    plt.xlabel("Missing Momentum")
    plt.legend()
    plt.yscale("log")
    if max(met)>300:
        plt.xlim(0,300)
    if save_plot_dir:
        plt.savefig(os.path.join(save_plot_dir,"missing_momentum.png"), dpi=dpi)
    if show_plot:
        plt.show()
    plt.close()




def jet_clustering(ojs, ptmin, pbar = True, num_of_jets = 1,  R=0.4):
#     print("clustering jets with paramert ptmin={}".format(ptmin))
    njets = []
    pTleadjets = [[] for _ in range(num_of_jets)]
    mleadjets = [[] for _ in range(num_of_jets)]
    for k in tqdm(range(len(ojs)), disable= not pbar):
        pseudojets_input = np.zeros(50, dtype=DTYPE_PTEPM)
        for i in range(50):
            pseudojets_input[i]['pT'] = ojs[k, i, 0]
            pseudojets_input[i]['eta'] = ojs[k, i, 1]
            pseudojets_input[i]['phi'] = ojs[k, i, 2]
        sequence = cluster(pseudojets_input, R=R, p=-1)
        jets = sequence.inclusive_jets(ptmin=ptmin)  # 5 gev
        njets += [len(jets)]
        for n_jet in range(num_of_jets):
            if (len(jets) > n_jet):
                pTleadjets[n_jet].append( jets[n_jet].pt)
                mleadjets[n_jet].append(jets[n_jet].mass)
    return njets, pTleadjets, mleadjets



# def _jet_clustering(ojs, ptmin, pbar = True):
# #     print("clustering jets with paramert ptmin={}".format(ptmin))
#     njets = []
#     pTleadjet = []
#     mleadjet = []
#     for k in tqdm(range(len(ojs)), disable= not pbar):
#         pseudojets_input = np.zeros(50, dtype=DTYPE_PTEPM)
#         for i in range(50):
#             pseudojets_input[i]['pT'] = ojs[k, i, 0]
#             pseudojets_input[i]['eta'] = ojs[k, i, 1]
#             pseudojets_input[i]['phi'] = ojs[k, i, 2]
#         sequence = cluster(pseudojets_input, R=0.1, p=-1)
#         jets = sequence.inclusive_jets(ptmin=ptmin)  # 5 gev
#         njets += [len(jets)]
#         if (len(jets) > 0):
#             pTleadjet += [jets[0].pt]
#             mleadjet += [jets[0].mass]
#     return njets, pTleadjet, mleadjet


def plot_njets(njets_summary, data_names, save_plot_dir, show_plot, dpi):
    plt.figure(figsize=_figsize)
    for i, njet in enumerate(njets_summary):
        if i==0:
            n,b,_= plt.hist(njet,label=data_names[i], alpha=0.5, histtype=_1st_hist_type, density=True, bins =20)
        else:
            n,b,_ = plt.hist(njet, label=data_names[i], bins=b, alpha=0.5, histtype=hist_type, density=True)
    plt.xlabel("n jets")
    plot_label = "n jets"
    plt.xlim(0,20)
    plt.ylim(0,0.5)
    plt.legend()
    if save_plot_dir:
        plt.savefig(os.path.join(save_plot_dir,plot_label.replace(" ","_")+".png"), dpi=dpi)
    if show_plot:
        plt.show()
    plt.close()
    
    
def plt_pt_m_jet(pt_or_m_summary, data_names, save_plot_dir, show_plot, dpi, jet_num, mode, mjet_xlim=None):
    plt.figure(figsize=_figsize)
    
    #check plot mode, 0 for pt jet, 1 for m jet
    name = ["pT jet", "m jet"][mode] 
    
    for i, data in enumerate(pt_or_m_summary):
        if i==0:
            n,b,_= plt.hist(data,label=data_names[i], alpha=0.5, histtype=_1st_hist_type, density=True, bins =np.logspace(1,3.5,20))
        else:
            n,b,_ = plt.hist(data, label=data_names[i], bins=b, alpha=0.5, histtype=hist_type, density=True)
    plot_label = name + " #{}".format(jet_num)
    plt.xlabel(plot_label)
    if mode==1 and not isinstance(mjet_xlim, type(None)):
        plt.xlim(*mjet_xlim)
    plt.semilogx()
    plt.yscale("log")
    plt.legend()
    if save_plot_dir:
        plt.savefig(os.path.join(save_plot_dir,plot_label.replace(" ","_")+".png"), dpi=dpi)
    if show_plot:
        plt.show()
    plt.close()


def plot_clustering(data_files, data_names, save_plot_dir=None, show_plot = True, pbar=True, num_of_jets = 1, 
                    dpi = 200, mjet_xlim = (10,100)):
    _ptmin = 10
    _bins = None
    njets_summary = []
    pt_summary=[]
    m_summary = []
    for df in data_files:
        njets, pTleadjets, mleadjets = jet_clustering(df,_ptmin, pbar, num_of_jets=num_of_jets)
        njets_summary.append(njets)
        pt_summary.append(pTleadjets)
        m_summary.append(mleadjets)
        
    plot_njets(njets_summary, data_names, save_plot_dir, show_plot, dpi)
    for jet_num in range(num_of_jets):
        pts = [data[jet_num] for data in pt_summary]
        ms = [data[jet_num] for data in m_summary]
        plt_pt_m_jet(pts, data_names, save_plot_dir, show_plot, dpi, jet_num, 0)
        plt_pt_m_jet(ms, data_names, save_plot_dir, show_plot, dpi, jet_num, 1, mjet_xlim)
    

def plot_everything(data_files, data_names, save_plot_dir=None, show_plot = True, pbar = True, njets = 1, dpi =200, mjet_xlim = (10,100)):
    plot_eta(data_files, data_names, save_plot_dir, show_plot, dpi)
    plot_phi(data_files, data_names, save_plot_dir, show_plot, dpi)
    plot_pt_frac(data_files, data_names, save_plot_dir, show_plot, dpi)
    plot_em(data_files, data_names, save_plot_dir, show_plot, pbar, dpi)
    plot_missing_m(data_files, data_names, save_plot_dir, show_plot, pbar, dpi)
    plot_clustering(data_files, data_names, save_plot_dir, show_plot, pbar, num_of_jets=njets, dpi=dpi, mjet_xlim=mjet_xlim)
    
    
def plot_recon_jet(data, log_betas, plot_root_dir, dpi = 200):
    for i,jet in enumerate(tqdm(data)):
        save_dir = os.path.join(plot_root_dir,"recon_plot_logbeta={:.2f}".format(log_betas[i]))
        try: 
            os.mkdir(save_dir) 
        except:
            pass
        plot_everything([jet,], ["recon_plot_logbeta={:.2f}".format(log_betas[i]),], save_plot_dir = save_dir, show_plot=False, pbar=False, dpi=dpi)
        

def compare_recons(original_data,data, log_betas, plot_root_dir, 
                   sub_dir_name="comparison", split=5, show_plot=True, njets=1, pbar = False, dpi = 200,
                  mjet_xlim = (10,100)):
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
    
    plot_everything(plot_entries, plot_name , save_plot_dir = save_dir, show_plot=show_plot, pbar=pbar, njets=njets, dpi=dpi, mjet_xlim=mjet_xlim)