import numpy as np
import pandas as pd
from scipy import stats


def win_ratio(model_A, model_B, method='welch'):
    # Calculate the win ratio between two models
    # print(np.array(model_A).shape, np.array(model_B).shape)
    stat_diff = []
    total_datasets = len(model_A)
    win_count = 0
    lose_count = 0
    for dataset in range(total_datasets):
        # Use a two-sample t-test to determine if there is a significant difference between the accuracy scores of the two models
        if method == 'welch':
            test, p = stats.ttest_ind(model_A[dataset], model_B[dataset], equal_var=False, random_state=0)
            # print(len(model_A[dataset]), len(model_B[dataset]), p)
        elif method == 'wilcoxon':
            diff = np.array(model_A[dataset]) - np.array(model_B[dataset])
            test, p = wilcoxon(diff) # returns sum of rank and pvalue
            
        if p < 0.05:
            stat_diff.append((dataset, test))
            
    # print('stat_diff', stat_diff)
    for s_diff in range(len(stat_diff)):
        count = 0
        
        results_a = model_A[stat_diff[s_diff][0]]
        results_b = model_B[stat_diff[s_diff][0]]
        
        # print('compare before loop', np.array(results_a).shape, np.array(results_b).shape)
        
        # Compare the accuracy scores of the two models for each dataset where there is a significant difference
        ## compares each fold result and counts how many folds model A is better than B 
        # for value1, value2 in zip(
        #     model_A[stat_diff[s_diff][0]], model_B[stat_diff[s_diff][0]]
        # ):
        #     print(value1, value2)
        #     if value1 > value2:
        #         count += 1
        # # Determine which model has more wins for each dataset where there is a significant difference
        # if count > len(model_A[stat_diff[s_diff][0]]) - count:
        #     win_count += 1
        
        if method == 'welch':
            mean_a, std_a = np.mean(results_a), np.std(results_a)
            mean_b, std_b = np.mean(results_b), np.std(results_b)

            # print (f' {mean_a} ({std_a}) vs {mean_b} ({std_b}); mean_a_better={mean_a > mean_b}; std_a_better={mean_a == mean_b and std_a < std_b}')

            # compare mean results of A and B
            if mean_a > mean_b:
                win_count += 1
            # if they are both equal, results with lower stdev wins
            elif mean_a == mean_b and std_a < std_b:
                win_count += 1
                
        elif method == 'wilcoxon':
            statistic = stat_diff[s_diff][1]
            # statistic is the sum of signed rank;
            # if this is positive, then results of method A is better than B
            if statistic>0:
                win_count += 1
        

    return win_count, len(stat_diff), stat_diff


def generator(results, models, method='welch'):
    ''' 
    assuming 2D matrix 
    (each row shows results across datasets for each model)
    '''
    # Generate accuracy scores for each dataset and model
    # Calculate the win ratio between each pair of models and store the results in a matrix
    cols, rows = len(results), len(results)
    
    output = [[0 for i in range(cols)] for j in range(rows)]
    
    for model_a in range(len(results)):
        for model_b in range(len(results)):
            if model_a!=model_b:
                out = list(win_ratio(results[model_a], results[model_b], method='welch'))
                output[model_a][model_b] = f"{out[0]}/{out[1]}"
                sig_diff_str = ', '.join([str(x[0]) for x in out[2]])
                # print(f'{models[model_a]} vs {models[model_b]}: sig_diff = [{sig_diff_str}]')
                # print(models[model_a], models[model_b], output[model_a][model_b])

    df = pd.DataFrame(output)
    df.index = models
    df.columns = models
    
    return df