import random
from helpers.corruptor_pipelines import get_col_preprocessor, get_fitted_pipeline, get_valid_pipeline
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statistics
import csv
import copy
import pickle

from sklearn.metrics import f1_score, accuracy_score, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler


from helpers.losses import my_info_nce
from tqdm import tqdm

from itertools import product, cycle
from functools import partial

from helpers.persistence import *
from helpers.progress_bar import ProgressBar
from helpers.data_utils import standardize_dataset, get_data_df
from helpers.models import UnsupervisedScarf, SupervisedScarf
from helpers.log_utils import *
# from helpers.openml_data import tabular_id_list, get_data, get_description_table
from helpers.openml_data_v2 import get_data, get_data1

from sklearn.model_selection import train_test_split

from helpers.corruptor import Corruptor
from time import time

default_settings = {
    'method': 'kfold',  # 'kfold' | 'bootstrap'

    'corruptor1': {
        'method': 'pass',
        
    },
    'corruptor2': {
        'method': 'pass',
    },
    # corruptor only drops first category if its OHEncoded
    # when doing corruption before OHE, corruptor will no drop because it doesnot know which cols are categorical
    # Please note, to keep dimensions consistent, both branches much follow these settings.
    'corrupt_before_ohe': False,  # only scarf use True 
    'scale_cat': False, 
    'kfolds': 5,
    'patience': 3,  # wait for 3 epochs to improve validation score; assuming validation is done every epoch
}

# only edit this from outside
settings = default_settings.copy()


device = "cuda:0" if torch.cuda.is_available() else "cpu"

pbar = False


def get_device():
    return device


# # legacy; not used in main functions
# k_folds = 5
# patience = 3
# y_column_name = 'target'


# def convert_df_to_tensor(df):
#     y_train = df[y_column_name]
#     X_train = df.drop([y_column_name], axis=1)
#     y_tensor = torch.tensor(y_train)
#     x_tensor = torch.tensor(X_train.values)
#     x_tensor = x_tensor.to(device)
#     y_tensor = y_tensor.to(device)

#     dataset_tensor = data_utils.TensorDataset(x_tensor, y_tensor)
#     return x_tensor, y_tensor, dataset_tensor


# overwrite; can handle nan
def average_multiple_lists(multiple_lists):
    # due to early stopping, the lists are now of different sizes i.e. ragged
    # we prefill them with nan and instead use mean/std functions that can handle nan values
    len_list = [len(x) for x in multiple_lists]
    max_len = max(len_list)
    tmp = [x + [np.NaN] * (max_len-len(x)) for x in multiple_lists]

    data = np.array(tmp)
    return np.nanmean(data, axis=0), np.nanstd(data, axis=0)


def model_pretraining(model, train_x, valid_x,
                      cols, cats, col_preprocessor,
                      num_of_epoch, batch_size,
                      save_path='./saved_vars/generated_default'):

    # to run on gpu
    model = model.to(device)
    # batch_size = len(dataset) # no batching
    pbar and pbar.set_description('preparing')

    data_loader = torch.utils.data.DataLoader(dataset=train_x,
                                              batch_size=batch_size,
                                              shuffle=True)
    
    # for training corruption
    # for training corruption
    preprocessor1 = get_fitted_pipeline(train_x, cols, cats, col_preprocessor,
                                        settings['corruptor1'],
                                        settings['corrupt_before_ohe'],
                                        settings['scale_cat'])
    preprocessor2 = get_fitted_pipeline(train_x, cols, cats, col_preprocessor,
                                        settings['corruptor2'],
                                        settings['corrupt_before_ohe'],
                                        settings['scale_cat'])
    # for validation corruption
    # valid_preprocessor1 = get_valid_pipeline(valid_x, preprocessor1)
    # valid_preprocessor2 = get_valid_pipeline(valid_x, preprocessor2)
    valid_preprocessor1 = preprocessor1
    valid_preprocessor2 = preprocessor2

    # replace the corruptor

    pbar.set_description('corrupting validation set')
    valid_dataloader_list = []
    # create corrupted validation datasets for 10 epochs
    # each validation set has a data loader
    for i in range(10):
        corupted_valid_x1 = valid_preprocessor1.transform(valid_x).float().to(device)
        corupted_valid_x2 = valid_preprocessor2.transform(valid_x).float().to(device)

        valid_dataset = data_utils.TensorDataset(
            corupted_valid_x1, corupted_valid_x2)
        valid_dataloader = data_utils.DataLoader(valid_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=True)
        valid_dataloader_list.append(valid_dataloader)

    # cycle through different validation dataset every 10 epochs
    valid_dataloader_cycler = cycle(valid_dataloader_list)
    ## print("After creating validation set: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))

    # latent is needed for stacked autoencoder pre-training
    latent = torch.tensor([])
    latent = latent.to(device)

    training_lle = []
    validation_lle = []
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    min_loss = np.inf
    best_state = model.state_dict()
    patience = settings['patience']
    tries = patience

    pbar and pbar.set_description('starting pretraining')

    for epoch in range(1, num_of_epoch+1):
        batch_losses = []
        batch_val_losses = []
        train_samples = 0
        valid_samples = 0
        found_better = False

        model = model.train()
        for batch_idx, train in enumerate(data_loader):

            train = train.float()
            train1 = preprocessor1.transform(train).float().to(device)
            train2 = preprocessor2.transform(train).float().to(device)

            assert train1.shape[1] == train2.shape[1], f'corrupted feature size different: {train1.shape} {train2.shape}'

            with torch.autograd.set_detect_anomaly(True):
                # optimizer.zero_grad()

                # noise in 2nd branch
                z1 = model(train1)
                z2 = model(train2)
                
                ## print('shapes z1, z2', z1.shape, ',', z2.shape)
                ## print("pretraining: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))

                loss = my_info_nce(z1, z2)  # contrastive loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_losses.append(loss.cpu().item() * train.shape[0])
            train_samples += train.shape[0]

        # validate the model after after each epoch
        # we need the val_loss per sample
        # val_loss_i = loss_i/2n
        # mean_val = sum_over(val_loss_i*2n) / (n_batch*2n)
        # or, mean_val = (sum_over(val_loss_i)*2n) / (n_batch*2n)
        # so, mean_val = sum_over(val_loss_i) / n_batch
        # however, samples in batches might not be same always

        model.eval()
        valid_data_loader = next(valid_dataloader_cycler)
        for batch_idx, (valid1, valid2) in enumerate(valid_data_loader):
            # validation data already preprocessed
            z1 = model(valid1)
            z2 = model(valid2)

            val_loss = my_info_nce(z1, z2)  # contrastive loss
            batch_val_losses.append(val_loss.cpu().item() * valid1.shape[0])
            valid_samples += valid1.shape[0]

        tries -= 1  # reduce tries after validation

        mean_loss = np.sum(batch_losses) / train_samples  # loss_per_sample
        mean_val_loss = np.sum(batch_val_losses) / \
            valid_samples  # val_loss_per_sample

        # save the model with minimum loss
        if mean_val_loss < min_loss:
            min_loss = mean_val_loss
            best_epoch = epoch
            best_state = model.state_dict()
            tries = patience  # replenish tries if we find improvement

        pbar and pbar.set_description((
            f'Pretraining | epoch: {epoch}/{num_of_epoch}; '
            f'mean_loss: {mean_loss:.3f}; '
            f'mean_valid_loss:{mean_val_loss:.3f}, min: {min_loss:.3f}; '
            f'tries: {tries}'
        ))
        # print(f'pretraining; mean_loss = {mean_loss}; mean_val_loss = {mean_val_loss};')

        # training_lle.append(loss.item())
        training_lle.append(mean_loss)
        validation_lle.append(mean_val_loss)

        # `patience` means no. of times to check
        # we stop if val_loss did not improve within tries
        if tries <= 0:
            break

    model.load_state_dict(best_state)

    return model, [training_lle, validation_lle], latent


def model_finetuning(model, preprocessor,
                     num_of_epoch, batch_size,
                     train_x, train_y,
                     valid_x, valid_y,
                     test_x, test_y):

    model = model.to(device)

    finetuning_loss = []
    finetuning_val_loss = []

    train_scores = []
    valid_scores = []
    test_scores = []

    # batch_size = len(train_tensor) # no batching
    # this preprocessing has no corruption as it is set to 'pass' mode
    
    dataset = data_utils.TensorDataset(train_x, train_y)
    data_loader = data_utils.DataLoader(dataset=dataset,
                                        batch_size=batch_size,
                                        shuffle=True)

    valid_dataset = data_utils.TensorDataset(valid_x, valid_y)
    valid_data_loader = data_utils.DataLoader(valid_dataset,
                                              batch_size=batch_size,
                                              shuffle=True)

    criterion = nn.CrossEntropyLoss()

    # lr=1e-5, weight_decay=1e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    patience = settings['patience']
    tries = patience
    min_loss = np.inf
    best_state = False

    # pbar = tqdm(range(1,num_of_epoch+1))
    for epoch in range(1, num_of_epoch+1):

        batch_losses = []
        batch_val_losses = []
        train_samples = 0
        valid_samples = 0
        found_better = False

        model = model.train()
        for batch_idx, (train, target) in enumerate(data_loader):
            optimizer.zero_grad()

            # to make sure all variables are float type
            train = preprocessor.transform(train).float().to(device)
            target = target.long()

            prediction = model(train)
            classification_loss = criterion(prediction, target)

            classification_loss.backward()
            optimizer.step()

            batch_losses.append(
                classification_loss.cpu().item() * train.shape[0])
            train_samples += train.shape[0]

        # valid every training epoch
        model = model.eval()
        for batch_idx, (valid, target) in enumerate(valid_data_loader):
            # to make sure all variables are float type
            valid = preprocessor.transform(valid).float().to(device)
            target = target.long()
            pred_valid = model(valid)
            valid_loss = criterion(pred_valid, target).item()

            batch_val_losses.append(valid_loss*valid.shape[0])
            valid_samples += valid.shape[0]
            
            # print(f'batch={batch_idx}; valid_loss={valid_loss}; valid.shape={valid.shape}')

        tries -= 1

        # mean loss per sample
        mean_loss = np.sum(batch_losses) / train_samples
        mean_val_loss = np.sum(batch_val_losses) / valid_samples
        
        # print(f'mean_loss = {mean_loss}; mean_val_loss = {mean_val_loss};')

        # save the model with minimum loss
        if mean_val_loss < min_loss:
            min_loss = mean_val_loss
            best_epoch = epoch
            best_state = model.state_dict()
            tries = patience

        pbar and pbar.set_description((
            f'Finetuning | epoch: {epoch}/{num_of_epoch};'
            f'mean_loss: {mean_loss:.3f} ; '
            f'mean_val_loss: {mean_val_loss:.3f} (min: {min_loss:.3f}); '
            f'tries = {tries}'
        ))

        finetuning_loss.append(mean_loss)
        finetuning_val_loss.append(mean_val_loss)

        # set to eval mode
        model = model.eval()

        y_train, y_valid, y_test = [x.cpu().detach().numpy() for x in [
            train_y, valid_y, test_y]]

        # train accuracy after each epoch
        pred_train = model(preprocessor.transform(train_x).float().to(device)
                           ).argmax(1).cpu().detach().numpy()
        train_accuracy = f1_score(y_train, pred_train, average='weighted')
        train_scores.append(train_accuracy)

        # test accuracy after each epoch
        pred_test = model(preprocessor.transform(
            test_x).float().to(device)).argmax(1).cpu().detach().numpy()
        test_accuracy = f1_score(y_test, pred_test, average='weighted')
        test_scores.append(test_accuracy)

        # valid accuracy after each epoch
        pred_valid = model(preprocessor.transform(valid_x).float().to(device)
                           ).argmax(1).cpu().detach().numpy()
        valid_score = f1_score(y_valid, pred_valid, average='weighted')
        valid_scores.append(valid_score)

        if tries <= 0:
            break

    model.load_state_dict(best_state)

    return model, [finetuning_loss, finetuning_val_loss], [train_scores, valid_scores, test_scores]


# todo: change to model_training
def ae_training(dataset_name, batch_size,
                unsupervised_epochs=1000,
                supervised_epochs=200
                ):

    df, y, cats = get_data1(dataset_name)
    cols = df.columns.values

    k_folds = settings['kfolds']
    # cv = KFold(n_splits = k_folds, random_state= 42, shuffle = True)
    cv = StratifiedKFold(n_splits=k_folds, random_state=42, shuffle=True)

    pretraining_loss_fold = []
    pretraining_valid_loss_fold = []
    finetuning_loss_fold = []
    finetuning_valid_loss_fold = []
    train_score_fold = []
    valid_score_fold = []
    test_score_fold = []
    final_valid_score_fold = []
    final_test_score_fold = []
    test_actuals = []
    test_predictions = []

    pbar and pbar.add_prefix('creating folds')
    fold_limit = 5 if settings['method'] == 'kfold' else 30

    kfold_splits = list(cv.split(df, y))
    
    # fit OHE before splitting
    col_preprocessor = get_col_preprocessor(cols, cats, settings['corrupt_before_ohe'])
    col_preprocessor.fit(df)

    # run k fold in loop
    for fold_counter in range(fold_limit):
        pbar and pbar.edit_last_prefix(
            f'fold={fold_counter+1}/{fold_limit} | ')

        if settings['method'] == 'kfold':
            train, test = kfold_splits[fold_counter]

            # divide the data for train and test
            train_df, train_y = df.iloc[train], y[train]
            test_df, test_y = df.iloc[test], y[test]
        else:
            train_df, test_df, train_y, test_y = train_test_split(df, y,
                                                                  stratify=y,
                                                                  shuffle=True,
                                                                  test_size=1/5,
                                                                  random_state=fold_counter)

        train_df, valid_df, train_y, valid_y = train_test_split(train_df, train_y,
                                                                stratify=train_y,
                                                                test_size=1/8, random_state=42)

        # we standardize/preprocess in pretraining and finetuning
        train_x, valid_x, test_x = [
            torch.from_numpy(x.values).float().to(device) for x in [train_df, valid_df, test_df]
        ]
        train_y, valid_y, test_y = [
            torch.from_numpy(x).long().to(device) for x in [train_y, valid_y, test_y]
        ]
        
        ## print("df to tensor: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))

        # preprocessor with corruptor in pass mode
        preprocessor = get_fitted_pipeline(train_x, cols, cats, col_preprocessor,
                                           default_settings['corruptor1'],
                                           settings['corrupt_before_ohe'],
                                           settings['scale_cat'])
        
        # cat_rescaler = preprocessor.named_steps['cat_rescaler'].named_transformers_['cat'].steps[0][1]
        # print('preprocessor')
        # print('mean', cat_rescaler.mean_)

        train_x_prepped = preprocessor.transform(train_x)
        x_dim = train_x_prepped.shape[1]
        # print(x_dim)
        output_size = torch.unique(train_y).shape[0]
        del train_x_prepped  # do not need this anymore
        
        ## print("x_dim calculation: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024), 'x_dim=', x_dim)

        # todo: rename to ssl_pretraining: x->f->g->z
        pretrain_model = UnsupervisedScarf(input_size=x_dim)
        ## print("pretraining model: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
        # pretraining done here
        # note: training_lle = (train_loss, valid_loss)
        # pretrained_model, training_lle, latent = model_pretraining(pretrain_model,
        #                                                            train_x, valid_x,
        #                                                            cols, cats, col_preprocessor,
        #                                                            unsupervised_epochs, batch_size)
        pretrained_model, training_lle, latent  = pretrain_model, [[], []], False
        ## print("after pretraining: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
        # copy the trained weights and biases when classifier is added to model
        # deep copy is recommended when copying weights and biases from a model
        encoder_sd = copy.deepcopy(pretrained_model.encoder.state_dict())

        model = SupervisedScarf(x_dim, output_size, pretrain_model.encoder)
        model.encoder.load_state_dict(encoder_sd)

        # finetuning done here
        # note: finetuning_loss = (train_loss, valid_loss)
        model, finetuning_loss, scores_list = model_finetuning(model, preprocessor,
                                                               supervised_epochs, batch_size,
                                                               train_x, train_y,
                                                               valid_x, valid_y,
                                                               test_x, test_y
                                                               )

        pretraining_loss_fold.append(training_lle[0])
        pretraining_valid_loss_fold.append(training_lle[1])
        finetuning_loss_fold.append(finetuning_loss[0])
        finetuning_valid_loss_fold.append(finetuning_loss[1])

        train_score_fold.append(scores_list[0])
        valid_score_fold.append(scores_list[1])
        test_score_fold.append(scores_list[2])

        def get_score(X_gpu, y_gpu):
            # calculate final test accuracy, confusion matrix etc
            X = X_gpu.float()
            y_true = y_gpu.cpu().detach().numpy()
            y_pred = model(preprocessor.transform(X).float().to(device)).cpu().detach().numpy().argmax(1)
            score = f1_score(y_true, y_pred, average='weighted')

            # plt.rcParams.update({'font.size': 13})
            # confusion_plt = ConfusionMatrixDisplay.from_predictions(y_true, y_pred,cmap=plt.cm.Greens)
            # plt.savefig(save_path+"_"+model_name+"_"+str(fold_counter))

            return score, y_true, y_pred

        valid_score, _, _ = get_score(valid_x, valid_y)
        test_score, y_true, y_pred = get_score(test_x, test_y)

        final_valid_score_fold.append(valid_score)
        final_test_score_fold.append(test_score)

        test_actuals.append(y_true)
        test_predictions.append(y_pred)

    # print(pretraining_loss_fold)
    # take average of fold losses
    training_lle = [
        average_multiple_lists(pretraining_loss_fold),
        average_multiple_lists(pretraining_valid_loss_fold)
    ]
    finetuning_loss = [
        average_multiple_lists(finetuning_loss_fold),
        average_multiple_lists(finetuning_valid_loss_fold)
    ]

    train_curve = average_multiple_lists(train_score_fold)
    valid_curve = average_multiple_lists(valid_score_fold)
    test_curve = average_multiple_lists(test_score_fold)

    classification_accuracy_mean = np.round(
        statistics.mean(final_test_score_fold), 4)
    classification_accuracy_std = np.round(
        statistics.pstdev(final_test_score_fold), 4)
    print("test accuracy:", classification_accuracy_mean,
          " (", classification_accuracy_std, ")")

    curve_list = [train_curve, valid_curve, test_curve]

    return (
        training_lle, finetuning_loss,
        curve_list, latent, final_test_score_fold,
        (  # loss values [n_fold x n_epoch] n_epoch varies
            pretraining_loss_fold, pretraining_valid_loss_fold,
            finetuning_loss_fold, finetuning_valid_loss_fold
        ),
        test_actuals, test_predictions
    )


def do_all(dataset_name='breast_cancer', batch_size=128,
           unsupervised_epochs=1000, supervised_epochs=200,
           expt='0'
           ):

    save_path = f"./generated/{expt}-"

    info = []

    starting_time = time()

    # training basic AE
    (pretraining_loss_list, finetuning_loss_list,
     curve_list, basic_latent, fold_scores, fold_curves,
     test_actuals, test_predictions) = ae_training(dataset_name,
                                                   batch_size=batch_size,
                                                   unsupervised_epochs=unsupervised_epochs,
                                                   supervised_epochs=supervised_epochs,
                                                   )

    time_taken = time() - starting_time
    n_folds = settings['kfolds'] if settings['method']=='kfold' else 30
    print(
        f'time taken, total={time_taken:.2f}s or {(time_taken/n_folds):.2f}s per fold')

    # dont need to save these results individually
    # print('Maximum MSELoss of training Basic: ',max(BLL_Error[0]))
    # print('Maximum BCELoss of finetuning: ',max(BFT_Error[0]))
    # # print('Mean accuracy: ', basic_accuracy)
    # info.append('Maximum MSELoss of Basic AE: ' + str(max(BLL_Error[0])) + '\n')
    # pd.DataFrame(basic_latent.cpu().detach().numpy()).to_csv(save_path + '_basic_latent'+ '.csv')
    # with open(save_path+'basic_pretrain.pkl', 'wb') as file:
    #     pickle.dump(BLL_Error, file)

    info_dict = {
        'trainer_settings': settings,
        'pretraining_loss_list': pretraining_loss_list,
        'finetuning_loss_list': finetuning_loss_list,
        'curve_list': curve_list,
        'basic_latent': basic_latent,  # not really needed to save/ not used
        'fold_scores': fold_scores,
        'fold_curves': fold_curves,
        'test_actuals': test_actuals,
        'test_predictions': test_predictions,
        'time_taken': time_taken,  # time in seconds
    }

    save_var(info_dict, save_path+'ae_training_returns.pkl')

    # shaded plots
#     shaded_plot(BLL_Error, "Basic AE Pretraining", 'Epoch', 'Loss', save_path + 'basic_pretraining.pdf')

#     shaded_plot(BFT_Error, "Basic AE Finetuning", 'Epoch', 'Loss', save_path + 'basic_finetuning.pdf')

    f1 = shaded_plot_multiple(pretraining_loss_list, ["train_loss", "valid_loss"],
                              'Epoch', 'Loss',
                              save_path + 'basic_pretraining.pdf')

    f2 = shaded_plot_multiple(finetuning_loss_list, ["train_loss", "valid_loss"],
                              'Epoch', 'Loss',
                              save_path + 'basic_finetuning.pdf')

    f3 = shaded_plot_multiple(curve_list, ["train_score", "valid_score", "test_score"],
                              'Epoch', 'Accuracy',
                              save_path + 'basic_accuracy.pdf')

    plot_each_fold(fold_curves, save_path+'ae_training.pdf', [f1, f2, f3])
    # file1 = open(save_path + ".txt","w")
    # file1.writelines(info)
    # file1.close()

    return fold_scores
