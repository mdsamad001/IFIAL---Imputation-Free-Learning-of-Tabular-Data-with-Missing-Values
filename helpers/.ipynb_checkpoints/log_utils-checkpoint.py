import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

def shaded_plot(pair, label, x_label, y_label, filename, show=False):
    mean = pair[0]
    std = pair[1]
    fig = plt.figure(figsize=(9, 6))
    x = np.arange(len(mean))
    plt.plot(x, mean, label=label)
    plt.fill_between(x, mean - std, mean + std, alpha=0.2)
    plt.legend(prop={'size': 13})
    plt.xlabel(x_label,fontsize=13)
    plt.ylabel(y_label,fontsize=13)
    plt.savefig(filename)
    plt.show() if show else plt.close()
    return fig


def shaded_plot_multiple(pairs, labels, x_label, y_label, filename, show=False):
    
    fig = plt.figure(figsize=(9, 6))

    for pair, label in zip(pairs, labels):
        mean = pair[0]
        std = pair[1]
        x = np.arange(len(mean))
        plt.plot(x, mean, label=label)
        plt.fill_between(x, mean - std, mean + std, alpha=0.2)
    
    plt.legend(prop={'size': 13})
    plt.xlabel(x_label,fontsize=13)
    plt.ylabel(y_label,fontsize=13)
    plt.savefig(filename)
    plt.show() if show else plt.close()
    return fig
    
    
def average_multiple_lists(multiple_lists):
    data = np.array(multiple_lists)
    return np.average(data, axis=0), np.std(data,axis=0)


def plot_each_fold(fold_curves, filename, add_figs = []):
    p_loss, pv_loss, f_loss, fv_loss = fold_curves
    
    with PdfPages(filename) as pdf:
        for f in add_figs:
            pdf.savefig(f)
            
        # plot pretraining loss per fold
        for fold_idx, (train_loss, val_loss) in enumerate(zip(p_loss, pv_loss)):
            fig = plt.figure(figsize=(9, 6))
            x = np.arange(len(train_loss))
            plt.plot(x, train_loss, label='train')
            plt.plot(x, val_loss, label='val')
            plt.xlabel('epoch',fontsize=13)
            plt.ylabel('mean contrastive loss',fontsize=13)
            plt.legend(prop={'size': 13})
            plt.title(f'Pretraining fold={fold_idx};')
            pdf.savefig(fig)
            plt.close()
            
        # plot finetuning loss per fold
        for fold_idx, (train_loss, val_loss) in enumerate(zip(f_loss, fv_loss)):
            fig = plt.figure(figsize=(9, 6))
            x = np.arange(len(train_loss))
            plt.plot(x, train_loss, label='train')
            plt.plot(x, val_loss, label='val')
            plt.xlabel('epoch',fontsize=13)
            plt.ylabel('mean cross-entropy loss',fontsize=13)
            plt.legend(prop={'size': 13})
            plt.title(f'Finetuning fold={fold_idx};')
            pdf.savefig(fig)
            plt.close()
            
            
        
        