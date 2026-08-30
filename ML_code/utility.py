import random
from sklearn.base import BaseEstimator, ClassifierMixin
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, LeaveOneOut, StratifiedKFold
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso, LogisticRegression, SGDClassifier
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import RFE, mutual_info_classif, SelectKBest, f_classif,chi2, mutual_info_classif
from sklearn.metrics import log_loss, accuracy_score
from mrmr import mrmr_classif
from sklearn.base import clone
import itertools
from sklearn.pipeline import Pipeline
import concurrent.futures 
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ProcessPoolExecutor
from sklearn.metrics import roc_auc_score
random.seed(42)
np.random.seed(42)


################# help functions #######################
def prep_omics(df):
    df = df.copy()
    # df.columns = [c.replace(":", ".") for c in df.columns]                           # normalize colons
    if "label" in df: df["label"] = df["label"].replace({"H":1, "N":0})              # encode label
    if "Demographic.Sex" in df: df["Demographic.Sex"] = df["Demographic.Sex"].replace({"Male":1,"Female":0})
    if "Demographic.Handedness" in df:                                                # one-hot handedness
        oh = pd.get_dummies(df["Demographic.Handedness"], prefix="Demographic.Handedness")
        df = pd.concat([df.drop(columns=["Demographic.Handedness"]), oh], axis=1)
    return df

def prepare_normalised_dataset(
        df,
        omics_features,             # proteomics + metabolomics
        numeric_demographic_cols,   # ONLY numeric demos
        keys,                       # ID columns to keep
        label_names,                # labels (kept unscaled)
        handedness_col="Demographic.Handedness",
        discrete_demo=["APOE4", "Demographic.Sex"]   # now a list
    ):
    
    df = df.copy()
    if handedness_col in df.columns:
        one_hot_hand = pd.get_dummies(df[handedness_col], prefix=handedness_col)
    else:
        one_hot_hand = pd.DataFrame(index=df.index)
    scale_cols = [c for c in omics_features if c in df.columns]
    scale_cols += [c for c in numeric_demographic_cols if c in df.columns]
    blocked = set(discrete_demo + keys + label_names + [handedness_col])
    scale_cols = [c for c in scale_cols if c not in blocked]

    # MinMax scaling
    scaler = MinMaxScaler()
    scaled = pd.DataFrame(
        scaler.fit_transform(df[scale_cols]),
        columns=scale_cols,
        index=df.index
    )
    keep_unscaled = df.drop(columns=scale_cols + [handedness_col], errors="ignore")
    final_df = pd.concat([scaled, keep_unscaled, one_hot_hand], axis=1)

    return final_df

def split_dataset_by_sex(df, feature_names, sex_col='Demographic.Sex'):
    """
    Split dataset into overall, male, and female subsets.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The full dataset
    feature_names : list
        List of feature column names to include (should contain 'label' and sex_col)
    sex_col : str, default='Demographic.Sex'
        Name of the sex/gender column (1=Male, 0=Female)
    
    Returns:
    --------
    tuple of (overall_df, male_df, female_df)
        - overall_df: DataFrame with all selected features including sex column
        - male_df: DataFrame with males only (sex column removed)
        - female_df: DataFrame with females only (sex column removed)
    
    Example:
    --------
    >>> baseline_fts = ['Age', 'BMI', 'Demographic.Sex', 'feature1', 'feature2']
    >>> overall, male, female = split_dataset_by_sex(df, baseline_fts + ['label'])
    """
    # Create a copy and select specified features
    overall_df = df.copy()
    overall_df = overall_df[feature_names]
    
    # Split by sex
    male_df = overall_df[overall_df[sex_col] == 1].copy()
    male_df = male_df.drop([sex_col], axis=1)
    
    female_df = overall_df[overall_df[sex_col] == 0].copy()
    female_df = female_df.drop([sex_col], axis=1)
    
    return overall_df, male_df, female_df



############# modelling functions #######################
# ============================================================================
# HELPER FUNCTION: Calculate Classification Metrics
# ============================================================================
def compute_classification_metrics(y_true, y_pred_prob, threshold='youden'):
    """
    Compute classification metrics at optimal threshold.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred_prob : array-like
        Predicted probabilities
    threshold : str or float, default='youden'
        'youden': Use Youden's Index (maximize sensitivity + specificity)
        float: Use fixed threshold value
    
    Returns:
    --------
    dict with auc, accuracy, sensitivity, specificity, precision, f1_score, threshold
    """
    y_true = np.array(y_true)
    y_pred_prob = np.array(y_pred_prob)
    
    # Calculate AUC first (threshold-independent)
    auc_score = roc_auc_score(y_true, y_pred_prob)
    
    if threshold == 'youden':
        # Find optimal threshold using Youden's Index
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
        youden_index = tpr - fpr
        optimal_idx = np.argmax(youden_index)
        optimal_threshold = thresholds[optimal_idx]
    else:
        optimal_threshold = threshold
    
    # Apply threshold
    y_pred = (y_pred_prob >= optimal_threshold).astype(int)
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall/TPR
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # TNR
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0    # PPV
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    
    return {
        'auc': auc_score,
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1_score': f1,
        'threshold': optimal_threshold
    }



def perform_feature_selection(method, params, model, X, y, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    if method == "rfe":
        print("Do RFE feature Selection")
        selector = RFE(model, n_features_to_select=params, step=50)
        selector.fit(X, y)
        selected_features = X.columns[selector.get_support(indices=True)]
    elif method == 'mrmr':
        print("Do MRMR feature selection")
        selected_features = mrmr_classif(X=X, y=y, K=params)
    elif method == 'select_k_best':
        print("Do SelectKBest feature selection")
        selector = SelectKBest(score_func=f_classif, k=params)
        selector.fit(X, y)
        selected_features = X.columns[selector.get_support(indices=True)]
    elif method == 'correlation':
        print("Do Correlation-Based feature selection")
        target_correlation = X.apply(lambda col: col.corr(pd.Series(y)))
        selected_features = target_correlation.abs().nlargest(params).index.tolist()
    elif method == 'mutual_info':
        print("Do Mutual Information feature selection")
        random_state = rng.integers(0, 100000)
        mi = mutual_info_classif(X, y, random_state=random_state)
        mi_series = pd.Series(mi, index=X.columns)
        selected_features = mi_series.nlargest(params).index.tolist()
    else:
        selected_features = X.columns 
    return selected_features

def _parallel_worker_task(run_seed, df, model_type, param_combinations, 
                          inner_n_split, outer_n_split, feature_selection_method, 
                          num_features):
    """
    Independent worker task that runs in its own process to ensure complete 
    memory and random state isolation.
    """
    # 1. Create a strictly local RNG for this specific run
    rng = np.random.default_rng(run_seed)
    
    # 2. Run the evaluation using the passed RNG
    (y_true, y_pred_prob, y_pred, feature_selection_df, avg_feature_count, 
     feature_coef_df, auc_scores, learning_curve, fold_metrics) = evaluate_model_with_stratified_cf(
        df=df, 
        model_type=model_type, 
        param_combinations=param_combinations, 
        inner_n_split=inner_n_split, 
        outer_n_split=outer_n_split,
        feature_selection_method=feature_selection_method, 
        num_features=num_features, 
        random_seed=run_seed, # Integer seed for StratifiedKFold
        rng=rng               # Generator for models
    )
    
    # 3. Package results
    result = {
        'run_seed': run_seed,
        'y_true': y_true,
        'y_pred_prob': y_pred_prob,
        'y_pred': y_pred,
        'feature_selection_df': feature_selection_df,
        'avg_feature_count': avg_feature_count,
        'feature_coef_df': feature_coef_df,
        'auc_scores_fold': auc_scores,
        'fold_metrics': fold_metrics
    }
    return result, y_true, y_pred_prob, auc_scores

# def parameter_tuning_cf(model_type, params, X_train, y_train, X_test, y_test, feature_selection_method=None, num_features = None, rng=None):
#     # Don't overwrite rng - use what's passed in!
#     if rng is None:
#         rng = np.random.default_rng(42)

#     if model_type == 'random_forest':
#         n_estimators, max_depth, min_samples_split, min_samples_leaf = params
#         model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
#                                        min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
#                                        class_weight='balanced', random_state=rng.integers(0, 100000) if rng else 42)
#     elif model_type == 'svm':
#         C, kernel, gamma = params
#         model = SVC(C=C, kernel=kernel, gamma=gamma, class_weight='balanced', probability=True, random_state=rng.integers(0, 100000) if rng else 42)
#     elif model_type == 'logistic_regression':
#         C, penalty, solver = params
#         if penalty == 'elasticnet' and solver != 'saga':
#             return
#         if penalty == 'elasticnet' and solver == 'saga':
#             model = LogisticRegression(C=C, penalty=penalty, l1_ratio=0.5, solver=solver, class_weight='balanced', random_state=rng.integers(0, 100000) if rng else 42)
#         else:
#             model = LogisticRegression(C=C, penalty=penalty, solver=solver, class_weight='balanced', random_state=rng.integers(0, 100000) if rng else 42)
#     elif model_type == 'sgd':
#         alpha, penalty, l1_ratio, loss = params
#         if penalty == 'elasticnet' or (penalty in ['l1', 'l2'] and l1_ratio == 0.15):
#             model = SGDClassifier(alpha=alpha, penalty=penalty, l1_ratio=l1_ratio, loss=loss,
#                                 class_weight='balanced', random_state=rng.integers(0, 100000) if rng else 42)
#         else:
#             return
#     # ADD THIS NEW CASE:
#     elif model_type == 'gradient_boosting':
#         n_estimators, max_depth, learning_rate, subsample = params
#         model = GradientBoostingClassifier(
#             n_estimators=n_estimators,
#             max_depth=max_depth,
#             learning_rate=learning_rate,
#             subsample=subsample,
#             random_state=rng.integers(0, 100000)
#         )
#     # REMOVE THIS CASE:
#     # elif model_type == 'qlattice':
#     #     n_epochs, max_complexity = params
#     #     model = QLatticeClassifier(n_epochs=n_epochs, max_complexity=max_complexity)

#     if feature_selection_method == "rfe":
#         selected_features = perform_feature_selection(feature_selection_method, num_features, model, X_train, y_train)
#         X_train, X_test = X_train[selected_features], X_test[selected_features]

#     # if model_type == 'qlattice':
#     #     train = pd.concat([X_train, y_train], axis=1)
#     #     model.fit(train)
#     #     test = pd.concat([X_test, y_test], axis=1)
#     #     y_pred_prob = model.predict_proba(test)
#     #     test_auc = roc_auc_score(y_test, y_pred_prob)

#     model.fit(X_train, y_train)
#     y_pred_prob = model.predict_proba(X_test)[:, 1]
#     test_auc = roc_auc_score(y_test, y_pred_prob)

#     return {
#         'model': model, 
#         'params': params, 
#         'test_auc': test_auc, 
#         'feature_selection_method': feature_selection_method,
#         'num_features': num_features, 
#         'selected_features':  selected_features if feature_selection_method == "rfe" else None
#     }

def parameter_tuning_cf(model_type, params, X_train, y_train, X_test, y_test, feature_selection_method=None, num_features = None, rng=None):
    # Ensure we have a generator
    if rng is None:
        rng = np.random.default_rng(42)
        
    # Generate a concrete integer seed for this specific model instance
    # This isolates the model's randomness from global state
    model_seed = rng.integers(0, 100000)

    if model_type == 'random_forest':
        n_estimators, max_depth, min_samples_split, min_samples_leaf = params
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                       min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                       class_weight='balanced', random_state=model_seed)
    elif model_type == 'svm':
        C, kernel, gamma = params
        model = SVC(C=C, kernel=kernel, gamma=gamma, class_weight='balanced', probability=True, random_state=model_seed)
    elif model_type == 'logistic_regression':
        C, penalty, solver = params
        if penalty == 'elasticnet' and solver != 'saga':
            return None
        if penalty == 'elasticnet' and solver == 'saga':
            model = LogisticRegression(C=C, penalty=penalty, l1_ratio=0.5, solver=solver, class_weight='balanced', random_state=model_seed)
        else:
            model = LogisticRegression(C=C, penalty=penalty, solver=solver, class_weight='balanced', random_state=model_seed)
    elif model_type == 'sgd':
        alpha, penalty, l1_ratio, loss = params
        if penalty == 'elasticnet' or (penalty in ['l1', 'l2'] and l1_ratio == 0.15):
            model = SGDClassifier(alpha=alpha, penalty=penalty, l1_ratio=l1_ratio, loss=loss,
                                class_weight='balanced', random_state=model_seed)
        else:
            return None
    elif model_type == 'gradient_boosting':
        n_estimators, max_depth, learning_rate, subsample = params
        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            random_state=model_seed
        )

    if feature_selection_method == "rfe":
        # Pass the model used for RFE (if it needs one)
        selected_features = perform_feature_selection(feature_selection_method, num_features, model, X_train, y_train, rng=rng)
        X_train, X_test = X_train[selected_features], X_test[selected_features]

    model.fit(X_train, y_train)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_pred_prob)
    
    return {
        'model': model, 
        'params': params, 
        'test_auc': test_auc, 
        'feature_selection_method': feature_selection_method,
        'num_features': num_features, 
        'selected_features':  selected_features if feature_selection_method == "rfe" else None
    }

def tune_model_parameters(model_name, param_combinations, X, y, n_splits=5, random_seed = 42, feature_selection_method=None, num_features = None, rng=None):
    results_list = []
    if rng is None:
        rng = np.random.default_rng(random_seed)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    for train_ix, test_ix in skf.split(X, y):
        X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
        y_train, y_test = y.iloc[train_ix], y.iloc[test_ix]
        if feature_selection_method and feature_selection_method != "rfe":
            selected_features = perform_feature_selection(feature_selection_method, num_features, None, X_train, y_train, rng=rng)
            X_train, X_test = X_train[selected_features], X_test[selected_features]

        for params in param_combinations:
            result = parameter_tuning_cf(model_name, params, X_train, y_train, X_test, y_test, feature_selection_method, num_features, rng=rng)
            if result is not None:
                results_list.append({'params': params, 'test_auc': result['test_auc']})
            

    results_df = pd.concat([pd.DataFrame([result]) for result in results_list], ignore_index=True)
    mean_auc_by_parameter = results_df.groupby('params')['test_auc'].mean().reset_index()
    best_parameters_row = mean_auc_by_parameter.sort_values(by='test_auc', ascending=False).iloc[0]
    # display(best_parameters_row)

    best_model = parameter_tuning_cf(model_name, best_parameters_row['params'], X, y, X, y, feature_selection_method, num_features, rng=rng)['model']
    
    return best_model, best_parameters_row


# def evaluate_model_with_stratified_cf(df, model_type, param_combinations, inner_n_split=5, outer_n_split=5, 
#                                       feature_selection_method=None, num_features=None, random_seed=42, rng=None):
#     """
#     Evaluate model with stratified cross-validation.
#     Now returns fold-level AUC and metrics consistently.
#     """
#     from sklearn.metrics import log_loss
    
#     if rng is None:
#         np.random.seed(random_seed)
#         random.seed(random_seed)
#         rng = np.random.default_rng(random_seed)


#     skf = StratifiedKFold(n_splits=outer_n_split, shuffle=True, random_state=random_seed)
#     y_true_list, y_pred_list, y_pred_prob_list = [], [], []
#     auc_scores = []  # AUC per fold
#     fold_metrics = []  # All metrics per fold
#     learning_curve = {'train_loss': [], 'test_loss': [], 
#                       'train_acc': [], 'test_acc': [],
#                       'train_auc': []}

#     X, y = df.drop('label', axis=1), df['label']
#     feature_names = X.columns.to_list()
#     feature_selection_count = {feature: 0 for feature in X.columns}
#     # feature_coefficient_sum = {feature: 0.0 for feature in X.columns} 
#     # total_feature_count = 0
#     # CHANGED: Track list of coefficients instead of sum
#     feature_coefficient_list = {feature: [] for feature in X.columns}
#     total_feature_count = 0
    

#     for train_val_index, test_index in skf.split(X, y):
#         X_train_val, X_test = X.iloc[train_val_index], X.iloc[test_index]
#         y_train_val, y_test = y.iloc[train_val_index], y.iloc[test_index]

#         ## Inner CV for hyperparameter tuning + feature selection
#         best_model, best_parameters_row = tune_model_parameters(
#             model_type, param_combinations, X_train_val, y_train_val, 
#             inner_n_split, feature_selection_method, num_features, rng=rng
#         )

#         ## Retrain on train_val with selected features
#         selected_features = perform_feature_selection(
#             feature_selection_method, num_features, best_model, X_train_val, y_train_val, rng=rng
#         )
#         X_train_selected = X_train_val.loc[:, selected_features]
#         X_test_selected = X_test.loc[:, selected_features]
        
#         # Train model
#         best_model.fit(X_train_selected, y_train_val)
#         y_test_pred = best_model.predict(X_test_selected).tolist()
#         y_test_pred_prob = best_model.predict_proba(X_test_selected)[:, 1].tolist()
#         y_train_pred_prob = best_model.predict_proba(X_train_selected)[:, 1].tolist()
#         y_train_pred = best_model.predict(X_train_selected).tolist()

#         # Feature importance extraction
#         if hasattr(best_model, 'coef_'):
#             # coefs = abs(best_model.coef_[0])
#             coefs = best_model.coef_[0]
#             for i, feature in enumerate(selected_features):
#                 # feature_coefficient_sum[feature] += coefs[i]
#                 feature_coefficient_list[feature].append(coefs[i])

#         if isinstance(best_model, LogisticRegression) and best_model.penalty == 'l1':
#             coefs = best_model.coef_[0]
#             selected_features = [feature_names[i] for i, coef in enumerate(coefs) if coef != 0]

#         if model_type in ['gradient_boosting', 'random_forest']:
#             importances = best_model.feature_importances_
#             for i, feature in enumerate(selected_features):
#                 # feature_coefficient_sum[feature] += importances[i]
#                 feature_coefficient_list[feature].append(importances[i])

#         if model_type == 'random_forest':
#             root_features = []
#             for estimator in best_model.estimators_:
#                 root_feature_index = estimator.tree_.feature[0]
#                 root_feature_name = X_train_selected.columns[root_feature_index]
#                 root_features.append(root_feature_name)
        
#         # Learning curve metrics
#         train_loss = log_loss(y_train_val, y_train_pred_prob)
#         test_loss = log_loss(y_test, y_test_pred_prob)
#         train_acc = accuracy_score(y_train_val, y_train_pred)
#         test_acc = accuracy_score(y_test, y_test_pred)
#         learning_curve['train_loss'].append(train_loss)
#         learning_curve['test_loss'].append(test_loss)
#         learning_curve['train_acc'].append(train_acc)
#         learning_curve['test_acc'].append(test_acc)
#         train_auc = roc_auc_score(y_train_val, y_train_pred_prob)
#         learning_curve['train_auc'].append(train_auc)
            
#         # OPTIMIZED: Calculate all metrics in single loop - O(n)
#         # compute_classification_metrics now includes AUC calculation
#         fold_metric = compute_classification_metrics(y_test.tolist(), y_test_pred_prob)
#         fold_metrics.append(fold_metric)
        
#         # Extract AUC for backward compatibility
#         fold_auc = fold_metric['auc']
#         auc_scores.append(fold_auc)
        
#         # Extend lists for overall metrics calculation
#         y_true_list.extend(y_test.tolist())
#         y_pred_list.extend(y_test_pred)
#         y_pred_prob_list.extend(y_test_pred_prob)

#         # Track feature selection
#         for ft in selected_features:
#             feature_selection_count[ft] += 1
#         total_feature_count += len(selected_features)

#     # Prepare Feature Selection DataFrame
#     feature_selection_df = pd.DataFrame(list(feature_selection_count.items()), columns=['Feature', 'Count'])
#     feature_selection_df = feature_selection_df.sort_values(by='Count', ascending=False)
#     feature_selection_df = feature_selection_df[feature_selection_df['Count'] > 0]
#     average_feature_count = total_feature_count / outer_n_split

#     # feature_coefficient_df = pd.DataFrame(list(feature_coefficient_sum.items()), columns=['Feature', 'CoefficientSum'])
#     # feature_coefficient_df = feature_coefficient_df.sort_values(by='CoefficientSum', ascending=False)
#     # feature_coefficient_df = feature_coefficient_df[feature_coefficient_df['CoefficientSum'] != 0]
#     # UPDATED: Calculate mean, std, and absolute mean for coefficients
#     feature_coefficient_data = []
#     for feature, coef_list in feature_coefficient_list.items():
#         if len(coef_list) > 0:  # Only include features selected at least once
#             feature_coefficient_data.append({
#                 'Feature': feature,
#                 'Mean_Coefficient': np.mean(coef_list),
#                 'Std_Coefficient': np.std(coef_list),
#                 'SE_Coefficient': np.std(coef_list) / np.sqrt(len(coef_list)),  # Standard Error
#                 'Median_Coefficient': np.median(coef_list),
#                 'Q1_Coefficient': np.percentile(coef_list, 25),
#                 'Q3_Coefficient': np.percentile(coef_list, 75),
#                 'Abs_Mean_Coefficient': np.mean(np.abs(coef_list)),  # For ranking
#                 'Count': len(coef_list)  # Number of times selected
#             })
    
#     feature_coefficient_df = pd.DataFrame(feature_coefficient_data)
#     feature_coefficient_df = feature_coefficient_df.sort_values(by='Abs_Mean_Coefficient', ascending=False)

#     return (y_true_list, y_pred_prob_list, y_pred_list, feature_selection_df, 
#             average_feature_count, feature_coefficient_df, auc_scores, learning_curve, fold_metrics)

def evaluate_model_with_stratified_cf(df, model_type, param_combinations, inner_n_split=5, outer_n_split=5, 
                                      feature_selection_method=None, num_features=None, random_seed=42, rng=None):
    from sklearn.metrics import log_loss
    
    # CRITICAL FIX: Do NOT set np.random.seed() here. 
    # Only create the local generator.
    if rng is None:
        rng = np.random.default_rng(random_seed)
        
    # Use random_seed integer for StratifiedKFold to ensure fold reproducibility
    skf = StratifiedKFold(n_splits=outer_n_split, shuffle=True, random_state=random_seed)
    
    y_true_list, y_pred_list, y_pred_prob_list = [], [], []
    auc_scores = []
    fold_metrics = []
    learning_curve = {'train_loss': [], 'test_loss': [], 
                      'train_acc': [], 'test_acc': [],
                      'train_auc': []}
                      
    X, y = df.drop('label', axis=1), df['label']
    feature_names = X.columns.to_list()
    feature_selection_count = {feature: 0 for feature in X.columns}
    feature_coefficient_list = {feature: [] for feature in X.columns}
    total_feature_count = 0
    
    for train_val_index, test_index in skf.split(X, y):
        X_train_val, X_test = X.iloc[train_val_index], X.iloc[test_index]
        y_train_val, y_test = y.iloc[train_val_index], y.iloc[test_index]
        
        best_model, best_parameters_row = tune_model_parameters(
            model_type, param_combinations, X_train_val, y_train_val, 
            inner_n_split, feature_selection_method, num_features, rng=rng
        )
        
        selected_features = perform_feature_selection(
            feature_selection_method, num_features, best_model, X_train_val, y_train_val, rng=rng
        )
        X_train_selected = X_train_val.loc[:, selected_features]
        X_test_selected = X_test.loc[:, selected_features]
        
        best_model.fit(X_train_selected, y_train_val)
        y_test_pred = best_model.predict(X_test_selected).tolist()
        y_test_pred_prob = best_model.predict_proba(X_test_selected)[:, 1].tolist()
        y_train_pred_prob = best_model.predict_proba(X_train_selected)[:, 1].tolist()
        y_train_pred = best_model.predict(X_train_selected).tolist()

        # ... (Rest of your feature importance and metric calculation logic remains exactly the same) ...
        # (Copied for brevity, ensure you keep your existing metric calculation code here)
        
        # Feature importance extraction. For L1 logistic regression, record only
        # genuinely selected (non-zero) coefficients. Previously all coefficients,
        # including zeros, were stored before the L1 filter and later appeared to
        # be "selected" in the aggregated coefficient table.
        if isinstance(best_model, LogisticRegression) and best_model.penalty == 'l1':
            coefs = best_model.coef_[0]
            fitted_features = list(selected_features)
            nonzero = np.flatnonzero(coefs != 0)
            selected_features = [fitted_features[i] for i in nonzero]
            for i in nonzero:
                feature_coefficient_list[fitted_features[i]].append(coefs[i])
        elif hasattr(best_model, 'coef_'):
            coefs = best_model.coef_[0]
            for i, feature in enumerate(selected_features):
                feature_coefficient_list[feature].append(coefs[i])
        if model_type in ['gradient_boosting', 'random_forest']:
            importances = best_model.feature_importances_
            for i, feature in enumerate(selected_features):
                feature_coefficient_list[feature].append(importances[i])
        
        train_loss = log_loss(y_train_val, y_train_pred_prob)
        test_loss = log_loss(y_test, y_test_pred_prob)
        train_acc = accuracy_score(y_train_val, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        learning_curve['train_loss'].append(train_loss)
        learning_curve['test_loss'].append(test_loss)
        learning_curve['train_acc'].append(train_acc)
        learning_curve['test_acc'].append(test_acc)
        train_auc = roc_auc_score(y_train_val, y_train_pred_prob)
        learning_curve['train_auc'].append(train_auc)
            
        fold_metric = compute_classification_metrics(y_test.tolist(), y_test_pred_prob)
        fold_metrics.append(fold_metric)
        auc_scores.append(fold_metric['auc'])
        
        y_true_list.extend(y_test.tolist())
        y_pred_list.extend(y_test_pred)
        y_pred_prob_list.extend(y_test_pred_prob)
        for ft in selected_features:
            feature_selection_count[ft] += 1
        total_feature_count += len(selected_features)

    # ... (Rest of your DataFrame creation logic remains the same) ...
    feature_selection_df = pd.DataFrame(list(feature_selection_count.items()), columns=['Feature', 'Count'])
    feature_selection_df = feature_selection_df.sort_values(by='Count', ascending=False)
    feature_selection_df = feature_selection_df[feature_selection_df['Count'] > 0]
    average_feature_count = total_feature_count / outer_n_split

    feature_coefficient_columns = [
        'Feature',
        'Mean_Coefficient',
        'Std_Coefficient',
        'SE_Coefficient',
        'Median_Coefficient',
        'Q1_Coefficient',
        'Q3_Coefficient',
        'Abs_Mean_Coefficient',
        'Count',
    ]
    feature_coefficient_data = []
    for feature, coef_list in feature_coefficient_list.items():
        if len(coef_list) > 0:
            feature_coefficient_data.append({
                'Feature': feature,
                'Mean_Coefficient': np.mean(coef_list),
                'Std_Coefficient': np.std(coef_list),
                'SE_Coefficient': np.std(coef_list) / np.sqrt(len(coef_list)),
                'Median_Coefficient': np.median(coef_list),
                'Q1_Coefficient': np.percentile(coef_list, 25),
                'Q3_Coefficient': np.percentile(coef_list, 75),
                'Abs_Mean_Coefficient': np.mean(np.abs(coef_list)),
                'Count': len(coef_list)
            })
    
    feature_coefficient_df = pd.DataFrame(
        feature_coefficient_data,
        columns=feature_coefficient_columns
    )
    feature_coefficient_df = feature_coefficient_df.sort_values(by='Abs_Mean_Coefficient', ascending=False)

    return (y_true_list, y_pred_prob_list, y_pred_list, feature_selection_df, 
            average_feature_count, feature_coefficient_df, auc_scores, learning_curve, fold_metrics)




# def run_multiple_evaluations_parallel(df, model_type, param_combinations, n_runs=10, inner_n_split=5, outer_n_split=5, 
#                                      feature_selection_method=None, num_features=None, global_seed=42):
#     """
#     Run multiple evaluations in parallel.
#     Returns consistent fold-level metrics (n_runs × outer_n_split entries).
#     """
#     all_auc_scores = []
#     all_y_true = []
#     all_y_pred_prob = []
#     all_results = []

#     np.random.seed(global_seed)
#     random.seed(global_seed)
#     seeds = [random.randint(0, 100000) for _ in range(n_runs)]

#     def evaluate_run(run_seed):
#         np.random.seed(run_seed)
#         random.seed(run_seed)
#         rng = np.random.default_rng(run_seed)
#         (y_true, y_pred_prob, y_pred, feature_selection_df, avg_feature_count, 
#          feature_coef_df, auc_scores, learning_curve, fold_metrics) = evaluate_model_with_stratified_cf(
#             df, model_type, param_combinations, inner_n_split=inner_n_split, outer_n_split=outer_n_split,
#             feature_selection_method=feature_selection_method, num_features=num_features, 
#             random_seed=run_seed, rng=rng
#         )
#         result = {
#             'run_seed': run_seed,
#             'y_true': y_true,
#             'y_pred_prob': y_pred_prob,
#             'y_pred': y_pred,
#             'feature_selection_df': feature_selection_df,
#             'avg_feature_count': avg_feature_count,
#             'feature_coef_df': feature_coef_df,
#             'auc_scores_fold': auc_scores,  # Per-fold AUC (5 values per run)
#             'fold_metrics': fold_metrics     # Per-fold metrics (5 dicts per run)
#         }
#         return result, y_true, y_pred_prob, auc_scores

#     # with ThreadPoolExecutor(max_workers=5) as executor:
#     #     futures = [executor.submit(evaluate_run, seed) for seed in seeds]
#     #     for future in futures:
#     #         result, y_true, y_pred_prob, auc_scores = future.result()
#     #         all_results.append(result)
#     #         all_y_true.extend(y_true)
#     #         all_y_pred_prob.extend(y_pred_prob)
#     #         all_auc_scores.extend(auc_scores)  # Extend with fold-level AUCs
#     with ThreadPoolExecutor(max_workers=5) as executor:
#         # Submit in order and collect in order
#         futures = {executor.submit(evaluate_run, seed): seed for seed in seeds}
#         for future in futures:  # This maintains order
#             result, y_true, y_pred_prob, auc_scores = future.result()
#             all_results.append(result)
#             all_y_true.extend(y_true)
#             all_y_pred_prob.extend(y_pred_prob)
#             all_auc_scores.extend(auc_scores)
#     # with ProcessPoolExecutor(max_workers=5) as executor:
#     #     futures = [executor.submit(evaluate_run, seed) for seed in seeds]
#     #     for future in futures:
#     #         result, y_true, y_pred_prob, auc_scores = future.result()
#     #         all_results.append(result)
#     #         all_y_true.extend(y_true)
#     #         all_y_pred_prob.extend(y_pred_prob)
#     #         all_auc_scores.extend(auc_scores)

#     # Calculate AUC statistics from FOLD-LEVEL scores
#     median_auc = np.median(all_auc_scores)
#     iqr_auc = stats.iqr(all_auc_scores)
#     auc_25th = np.percentile(all_auc_scores, 25)
#     auc_75th = np.percentile(all_auc_scores, 75)
#     overall_auc = roc_auc_score(all_y_true, all_y_pred_prob)

#     # Calculate overall classification metrics
#     overall_metrics = compute_classification_metrics(all_y_true, all_y_pred_prob)
    
#     # Extract FOLD-LEVEL metrics for median and IQR calculation
#     all_accuracy = []
#     all_sensitivity = []
#     all_specificity = []
#     all_precision = []
#     all_f1 = []
    
#     for result in all_results:
#         for fold_metric in result['fold_metrics']:
#             all_accuracy.append(fold_metric['accuracy'])
#             all_sensitivity.append(fold_metric['sensitivity'])
#             all_specificity.append(fold_metric['specificity'])
#             all_precision.append(fold_metric['precision'])
#             all_f1.append(fold_metric['f1_score'])
    
#     # Calculate median and IQR for each metric
#     metrics_summary = {
#         'auc': {
#             'overall': overall_auc,
#             'median': median_auc,
#             'iqr': iqr_auc,
#             'q1': auc_25th,
#             'q3': auc_75th
#         },
#         'accuracy': {
#             'overall': overall_metrics['accuracy'],
#             'median': np.median(all_accuracy),
#             'iqr': stats.iqr(all_accuracy),
#             'q1': np.percentile(all_accuracy, 25),
#             'q3': np.percentile(all_accuracy, 75)
#         },
#         'sensitivity': {
#             'overall': overall_metrics['sensitivity'],
#             'median': np.median(all_sensitivity),
#             'iqr': stats.iqr(all_sensitivity),
#             'q1': np.percentile(all_sensitivity, 25),
#             'q3': np.percentile(all_sensitivity, 75)
#         },
#         'specificity': {
#             'overall': overall_metrics['specificity'],
#             'median': np.median(all_specificity),
#             'iqr': stats.iqr(all_specificity),
#             'q1': np.percentile(all_specificity, 25),
#             'q3': np.percentile(all_specificity, 75)
#         },
#         'precision': {
#             'overall': overall_metrics['precision'],
#             'median': np.median(all_precision),
#             'iqr': stats.iqr(all_precision),
#             'q1': np.percentile(all_precision, 25),
#             'q3': np.percentile(all_precision, 75)
#         },
#         'f1_score': {
#             'overall': overall_metrics['f1_score'],
#             'median': np.median(all_f1),
#             'iqr': stats.iqr(all_f1),
#             'q1': np.percentile(all_f1, 25),
#             'q3': np.percentile(all_f1, 75)
#         }
#     }

#     return all_results, overall_auc, median_auc, iqr_auc, (auc_25th, auc_75th), metrics_summary

# def run_multiple_evaluations_parallel(df, model_type, param_combinations, n_runs=10, inner_n_split=5, outer_n_split=5, 
#                                      feature_selection_method=None, num_features=None, global_seed=42):
    
#     all_auc_scores = []
#     all_y_true = []
#     all_y_pred_prob = []
#     all_results = []
    
#     # 1. Set global seed ONCE in the main thread to generate the list of seeds
#     np.random.seed(global_seed)
#     random.seed(global_seed)
    
#     # Generate deterministic seeds for the children
#     seeds = [random.randint(0, 100000) for _ in range(n_runs)]

#     def evaluate_run(run_seed):
#         # CRITICAL FIX: Removed np.random.seed(run_seed) and random.seed(run_seed)
#         # We rely strictly on the local rng and the passed run_seed
#         rng = np.random.default_rng(run_seed)
        
#         (y_true, y_pred_prob, y_pred, feature_selection_df, avg_feature_count, 
#          feature_coef_df, auc_scores, learning_curve, fold_metrics) = evaluate_model_with_stratified_cf(
#             df, model_type, param_combinations, inner_n_split=inner_n_split, outer_n_split=outer_n_split,
#             feature_selection_method=feature_selection_method, num_features=num_features, 
#             random_seed=run_seed, rng=rng 
#         )
        
#         result = {
#             'run_seed': run_seed,
#             'y_true': y_true,
#             'y_pred_prob': y_pred_prob,
#             'y_pred': y_pred,
#             'feature_selection_df': feature_selection_df,
#             'avg_feature_count': avg_feature_count,
#             'feature_coef_df': feature_coef_df,
#             'auc_scores_fold': auc_scores,
#             'fold_metrics': fold_metrics
#         }
#         return result, y_true, y_pred_prob, auc_scores

#     # Execute
#     with ThreadPoolExecutor(max_workers=5) as executor:
#         futures = {executor.submit(evaluate_run, seed): seed for seed in seeds}
#         for future in futures:
#             result, y_true, y_pred_prob, auc_scores = future.result()
#             all_results.append(result)
#             all_y_true.extend(y_true)
#             all_y_pred_prob.extend(y_pred_prob)
#             all_auc_scores.extend(auc_scores)

#     # ... (Rest of the metrics summary logic remains the same) ...
#     # Calculate AUC statistics from FOLD-LEVEL scores
#     median_auc = np.median(all_auc_scores)
#     iqr_auc = stats.iqr(all_auc_scores)
#     auc_25th = np.percentile(all_auc_scores, 25)
#     auc_75th = np.percentile(all_auc_scores, 75)
#     overall_auc = roc_auc_score(all_y_true, all_y_pred_prob)
    
#     overall_metrics = compute_classification_metrics(all_y_true, all_y_pred_prob)
    
#     all_accuracy = []
#     all_sensitivity = []
#     all_specificity = []
#     all_precision = []
#     all_f1 = []
    
#     for result in all_results:
#         for fold_metric in result['fold_metrics']:
#             all_accuracy.append(fold_metric['accuracy'])
#             all_sensitivity.append(fold_metric['sensitivity'])
#             all_specificity.append(fold_metric['specificity'])
#             all_precision.append(fold_metric['precision'])
#             all_f1.append(fold_metric['f1_score'])
    
#     metrics_summary = {
#         'auc': {'overall': overall_auc, 'median': median_auc, 'iqr': iqr_auc, 'q1': auc_25th, 'q3': auc_75th},
#         'accuracy': {'overall': overall_metrics['accuracy'], 'median': np.median(all_accuracy), 'iqr': stats.iqr(all_accuracy), 'q1': np.percentile(all_accuracy, 25), 'q3': np.percentile(all_accuracy, 75)},
#         'sensitivity': {'overall': overall_metrics['sensitivity'], 'median': np.median(all_sensitivity), 'iqr': stats.iqr(all_sensitivity), 'q1': np.percentile(all_sensitivity, 25), 'q3': np.percentile(all_sensitivity, 75)},
#         'specificity': {'overall': overall_metrics['specificity'], 'median': np.median(all_specificity), 'iqr': stats.iqr(all_specificity), 'q1': np.percentile(all_specificity, 25), 'q3': np.percentile(all_specificity, 75)},
#         'precision': {'overall': overall_metrics['precision'], 'median': np.median(all_precision), 'iqr': stats.iqr(all_precision), 'q1': np.percentile(all_precision, 25), 'q3': np.percentile(all_precision, 75)},
#         'f1_score': {'overall': overall_metrics['f1_score'], 'median': np.median(all_f1), 'iqr': stats.iqr(all_f1), 'q1': np.percentile(all_f1, 25), 'q3': np.percentile(all_f1, 75)}
#     }
    
#     return all_results, overall_auc, median_auc, iqr_auc, (auc_25th, auc_75th), metrics_summary





def run_multiple_evaluations_parallel(df, model_type, param_combinations, n_runs=10, 
                                     inner_n_split=5, outer_n_split=5, 
                                     feature_selection_method=None, num_features=None, 
                                     global_seed=42):
    
    all_auc_scores = []
    all_y_true = []
    all_y_pred_prob = []
    all_results = []
    
    # 1. Set global seed ONCE in the main process to generate the list of seeds
    np.random.seed(global_seed)
    random.seed(global_seed)
    
    # Generate deterministic seeds for the children
    seeds = [random.randint(0, 100000) for _ in range(n_runs)]

    # 2. Use ProcessPoolExecutor for memory isolation
    # We use a list to keep track of futures to ensure we can debug order if needed,
    # though as_completed or direct iteration is fine.
    with ProcessPoolExecutor(max_workers=5) as executor:
        # Submit tasks
        # We pass all necessary data explicitly since the worker is now global
        futures = [
            executor.submit(
                _parallel_worker_task, 
                seed, 
                df, 
                model_type, 
                param_combinations, 
                inner_n_split, 
                outer_n_split, 
                feature_selection_method, 
                num_features
            ) for seed in seeds
        ]
        
        # Collect results
        for future in futures:
            result, y_true, y_pred_prob, auc_scores = future.result()
            all_results.append(result)
            all_y_true.extend(y_true)
            all_y_pred_prob.extend(y_pred_prob)
            all_auc_scores.extend(auc_scores)

    # 3. Calculate AUC statistics from FOLD-LEVEL scores
    median_auc = np.median(all_auc_scores)
    iqr_auc = stats.iqr(all_auc_scores)
    auc_25th = np.percentile(all_auc_scores, 25)
    auc_75th = np.percentile(all_auc_scores, 75)
    overall_auc = roc_auc_score(all_y_true, all_y_pred_prob)
    
    overall_metrics = compute_classification_metrics(all_y_true, all_y_pred_prob)
    
    all_accuracy = []
    all_sensitivity = []
    all_specificity = []
    all_precision = []
    all_f1 = []
    
    for result in all_results:
        for fold_metric in result['fold_metrics']:
            all_accuracy.append(fold_metric['accuracy'])
            all_sensitivity.append(fold_metric['sensitivity'])
            all_specificity.append(fold_metric['specificity'])
            all_precision.append(fold_metric['precision'])
            all_f1.append(fold_metric['f1_score'])
    
    metrics_summary = {
        'auc': {
            'overall': overall_auc, 
            'median': median_auc, 
            'iqr': iqr_auc, 
            'q1': auc_25th, 
            'q3': auc_75th
        },
        'accuracy': {
            'overall': overall_metrics['accuracy'], 
            'median': np.median(all_accuracy), 
            'iqr': stats.iqr(all_accuracy), 
            'q1': np.percentile(all_accuracy, 25), 
            'q3': np.percentile(all_accuracy, 75)
        },
        'sensitivity': {
            'overall': overall_metrics['sensitivity'], 
            'median': np.median(all_sensitivity), 
            'iqr': stats.iqr(all_sensitivity), 
            'q1': np.percentile(all_sensitivity, 25), 
            'q3': np.percentile(all_sensitivity, 75)
        },
        'specificity': {
            'overall': overall_metrics['specificity'], 
            'median': np.median(all_specificity), 
            'iqr': stats.iqr(all_specificity), 
            'q1': np.percentile(all_specificity, 25), 
            'q3': np.percentile(all_specificity, 75)
        },
        'precision': {
            'overall': overall_metrics['precision'], 
            'median': np.median(all_precision), 
            'iqr': stats.iqr(all_precision), 
            'q1': np.percentile(all_precision, 25), 
            'q3': np.percentile(all_precision, 75)
        },
        'f1_score': {
            'overall': overall_metrics['f1_score'], 
            'median': np.median(all_f1), 
            'iqr': stats.iqr(all_f1), 
            'q1': np.percentile(all_f1, 25), 
            'q3': np.percentile(all_f1, 75)
        }
    }
    
    return all_results, overall_auc, median_auc, iqr_auc, (auc_25th, auc_75th), metrics_summary
















# def run_multiple_evaluations_parallel(df, model_type, param_combinations, n_runs=10, 
#                                      inner_n_split=5, outer_n_split=5, 
#                                      feature_selection_method=None, num_features=None, 
#                                      global_seed=42):
    
#     all_auc_scores = []
#     all_y_true = []
#     all_y_pred_prob = []
#     all_results = []
    
#     # 1. Generate the list of seeds deterministically
#     np.random.seed(global_seed)
#     random.seed(global_seed)
#     seeds = [random.randint(0, 100000) for _ in range(n_runs)]

#     # 2. Use ProcessPoolExecutor for memory isolation
#     # Note: max_workers is usually set to CPU count, but keep it manageable (e.g., 5)
#     with ProcessPoolExecutor(max_workers=5) as executor:
#         # Submit tasks securely keeping the order
#         futures = []
#         for seed in seeds:
#             future = executor.submit(
#                 _parallel_worker_task, # Calling the global helper
#                 seed, 
#                 df, 
#                 model_type, 
#                 param_combinations, 
#                 inner_n_split, 
#                 outer_n_split, 
#                 feature_selection_method, 
#                 num_features
#             )
#             futures.append(future)

#         # Collect results IN ORDER
#         for future in futures:
#             result, y_true, y_pred_prob, auc_scores = future.result()
#             all_results.append(result)
#             all_y_true.extend(y_true)
#             all_y_pred_prob.extend(y_pred_prob)
#             all_auc_scores.extend(auc_scores)

#     # 3. Calculate statistics (Same as your original code)
#     median_auc = np.median(all_auc_scores)
#     iqr_auc = stats.iqr(all_auc_scores)
#     auc_25th = np.percentile(all_auc_scores, 25)
#     auc_75th = np.percentile(all_auc_scores, 75)
    
#     # Note: Calculating overall AUC on concatenated probs can be biased if folds are unbalanced
#     # but strictly for your code structure, we keep it:
#     overall_auc = roc_auc_score(all_y_true, all_y_pred_prob)
#     overall_metrics = compute_classification_metrics(all_y_true, all_y_pred_prob)
    
#     all_accuracy = []
#     all_sensitivity = []
#     all_specificity = []
#     all_precision = []
#     all_f1 = []
    
#     for result in all_results:
#         for fold_metric in result['fold_metrics']:
#             all_accuracy.append(fold_metric['accuracy'])
#             all_sensitivity.append(fold_metric['sensitivity'])
#             all_specificity.append(fold_metric['specificity'])
#             all_precision.append(fold_metric['precision'])
#             all_f1.append(fold_metric['f1_score'])
    
#     metrics_summary = {
#         'auc': {'overall': overall_auc, 'median': median_auc, 'iqr': iqr_auc, 'q1': auc_25th, 'q3': auc_75th},
#         'accuracy': {'overall': overall_metrics['accuracy'], 'median': np.median(all_accuracy), 'iqr': stats.iqr(all_accuracy), 'q1': np.percentile(all_accuracy, 25), 'q3': np.percentile(all_accuracy, 75)},
#         'sensitivity': {'overall': overall_metrics['sensitivity'], 'median': np.median(all_sensitivity), 'iqr': stats.iqr(all_sensitivity), 'q1': np.percentile(all_sensitivity, 25), 'q3': np.percentile(all_sensitivity, 75)},
#         'specificity': {'overall': overall_metrics['specificity'], 'median': np.median(all_specificity), 'iqr': stats.iqr(all_specificity), 'q1': np.percentile(all_specificity, 25), 'q3': np.percentile(all_specificity, 75)},
#         'precision': {'overall': overall_metrics['precision'], 'median': np.median(all_precision), 'iqr': stats.iqr(all_precision), 'q1': np.percentile(all_precision, 25), 'q3': np.percentile(all_precision, 75)},
#         'f1_score': {'overall': overall_metrics['f1_score'], 'median': np.median(all_f1), 'iqr': stats.iqr(all_f1), 'q1': np.percentile(all_f1, 25), 'q3': np.percentile(all_f1, 75)}
#     }
    
#     return all_results, overall_auc, median_auc, iqr_auc, (auc_25th, auc_75th), metrics_summary

def extract_results(all_results, key):
    """
    Extract specific results from all_results.
    Now properly handles fold-level metrics.
    """
    extracted_data = []
    
    if key == 'feature_selection_df':
        feature_counts = {}
        for result in all_results:
            for feature, count in result['feature_selection_df'].values:
                if feature in feature_counts:
                    feature_counts[feature] += count
                else:
                    feature_counts[feature] = count
        extracted_data = pd.DataFrame(list(feature_counts.items()), columns=['Feature', 'Total Count'])
        extracted_data = extracted_data.sort_values('Total Count', ascending=False)
        
    # elif key == 'feature_coef_df':
    #     coef_sums = {}
    #     for result in all_results:
    #         for feature, coef_sum in result['feature_coef_df'].values:
    #             if feature in coef_sums:
    #                 coef_sums[feature] += coef_sum
    #             else:
    #                 coef_sums[feature] = coef_sum
    #     extracted_data = pd.DataFrame(list(coef_sums.items()), columns=['Feature', 'Coefficient Sum'])
    #     extracted_data = extracted_data.sort_values('Coefficient Sum', ascending=False)
    elif key == 'feature_coef_df':
        # UPDATED: Aggregate coefficient statistics across all runs
        feature_coef_aggregated = {}
        
        for result in all_results:
            for _, row in result['feature_coef_df'].iterrows():
                feature = row['Feature']
                if feature not in feature_coef_aggregated:
                    feature_coef_aggregated[feature] = {
                        'coef_values': [],
                        'abs_coef_values': []
                    }
                
                # Store the mean coefficient from this run
                feature_coef_aggregated[feature]['coef_values'].append(row['Mean_Coefficient'])
                feature_coef_aggregated[feature]['abs_coef_values'].append(row['Abs_Mean_Coefficient'])
        
        coefficient_columns = [
            'Feature',
            'Mean_Coefficient',
            'Std_Coefficient',
            'SE_Coefficient',
            'Median_Coefficient',
            'Q1_Coefficient',
            'Q3_Coefficient',
            'Abs_Mean_Coefficient',
            'Count',
        ]

        # Calculate overall statistics
        coef_summary = []
        for feature, data in feature_coef_aggregated.items():
            coef_summary.append({
                'Feature': feature,
                'Mean_Coefficient': np.mean(data['coef_values']),
                'Std_Coefficient': np.std(data['coef_values']),
                'SE_Coefficient': np.std(data['coef_values']) / np.sqrt(len(data['coef_values'])),
                'Median_Coefficient': np.median(data['coef_values']),
                'Q1_Coefficient': np.percentile(data['coef_values'], 25),
                'Q3_Coefficient': np.percentile(data['coef_values'], 75),
                'Abs_Mean_Coefficient': np.mean(data['abs_coef_values']),
                'Count': len(data['coef_values'])
            })
        
        extracted_data = pd.DataFrame(coef_summary, columns=coefficient_columns)
        extracted_data = extracted_data.sort_values('Abs_Mean_Coefficient', ascending=False)

        
    elif key == 'fold_metrics':
        # Return all fold metrics (n_runs × outer_n_split dictionaries)
        for result in all_results:
            extracted_data.extend(result['fold_metrics'])
            
    elif key == 'auc_scores_fold':
        # Return fold-level AUC scores (n_runs × outer_n_split values)
        for result in all_results:
            extracted_data.extend(result['auc_scores_fold'])
            
    elif key in ['y_true', 'y_pred_prob', 'y_pred']:
        for result in all_results:
            extracted_data.extend(result[key])
            
    return extracted_data


# def evaluate_full_dataset(df, model_type, param_combinations, feature_selection_method=None, num_features=None):
#     X, y = df.drop('label', axis=1), df['label']
#     feature_names = X.columns.to_list()
#     feature_coefficient_sum = {feature: 0.0 for feature in X.columns}

#     # Tune model parameters using the entire dataset
#     best_model, best_parameters_row = tune_model_parameters(model_type, param_combinations, X, y, n_splits=5, feature_selection_method=feature_selection_method, num_features=num_features)

#     # Perform feature selection if specified
#     selected_features = perform_feature_selection(feature_selection_method, num_features, best_model, X, y)
#     X_selected = X.loc[:, selected_features]


#     # Fit the model on the entire dataset
    
#     best_model.fit(X_selected, y)
#     y_pred = best_model.predict(X_selected).tolist()
#     y_pred_prob = best_model.predict_proba(X_selected)[:, 1].tolist()

#     # Extract feature coefficients if available
#     if hasattr(best_model, 'coef_'):
#         coefs = best_model.coef_[0]  ########## add the absolute value here
#         for i, feature in enumerate(selected_features):
#             feature_coefficient_sum[feature] += coefs[i]
            
#     if isinstance(best_model, LogisticRegression) and best_model.penalty == 'l1':
#         coefs = best_model.coef_[0]
#         selected_features = [feature_names[i] for i, coef in enumerate(coefs) if coef != 0]

#     if model_type == 'random_forest':
#         root_features = []
#         for estimator in best_model.estimators_:
#             root_feature_index = estimator.tree_.feature[0]
#             root_feature_name = X_selected.columns[root_feature_index]
#             root_features.append(root_feature_name)
#     if model_type == 'gradient_boosting':
#         importances = best_model.feature_importances_
#         for i, feature in enumerate(selected_features):
#             feature_coefficient_sum[feature] += importances[i]

#     # Prepare feature coefficient DataFrame
#     feature_coefficient_df = pd.DataFrame(list(feature_coefficient_sum.items()), columns=['Feature', 'CoefficientSum'])
#     feature_coefficient_df = feature_coefficient_df.sort_values(by='CoefficientSum', ascending=False)
#     feature_coefficient_df = feature_coefficient_df[feature_coefficient_df['CoefficientSum'] != 0]

#     return y.tolist(), y_pred_prob, y_pred, feature_coefficient_df

# def evaluate_10_splits(df, model_type, param_combinations, feature_selection_method=None, num_features=None, global_seed=42, rng=None):
#     auc_scores = []
#     np.random.seed(global_seed)
#     rng = np.random.default_rng(global_seed)

#     # Perform the 10 splits
#     for _ in range(10):
#         # Split dataset into training and test sets (60-40 split)
#         train_df, test_df = train_test_split(df, test_size=0.4, random_state=rng.integers(0, 10000))
#         # train_df, test_df = train_test_split(df, test_size=0.4, random_state=np.random.randint(10000))
#         X_train, y_train = train_df.drop('label', axis=1), train_df['label']
#         X_test, y_test = test_df.drop('label', axis=1), test_df['label']
#         feature_names = X_train.columns.to_list()
#         feature_coefficient_sum = {feature: 0.0 for feature in X_train.columns}

#         # Tune model parameters using the training dataset only
#         best_model, best_parameters_row = tune_model_parameters(
#             model_type, param_combinations, X_train, y_train, n_splits=5, 
#             feature_selection_method=feature_selection_method, num_features=num_features
#         )

#         # Perform feature selection if specified
#         selected_features = perform_feature_selection(feature_selection_method, num_features, best_model, X_train, y_train)
#         X_train_selected = X_train.loc[:, selected_features]
#         X_test_selected = X_test.loc[:, selected_features]

#         # Fit the model on the training dataset
#         best_model.fit(X_train_selected, y_train)
#         y_pred_prob = best_model.predict_proba(X_test_selected)[:, 1]

#         # Calculate AUC and store the result
#         auc_score = roc_auc_score(y_test, y_pred_prob)
#         auc_scores.append(auc_score)

#         # Extract feature coefficients if available
#         if hasattr(best_model, 'coef_'):
#             # coefs = abs(best_model.coef_[0])  # Using absolute value of coefficients
#             coefs = best_model.coef_[0]  
#             for i, feature in enumerate(selected_features):
#                 feature_coefficient_sum[feature] += coefs[i]

#         if isinstance(best_model, LogisticRegression) and best_model.penalty == 'l1':
#             coefs = best_model.coef_[0]
#             selected_features = [feature_names[i] for i, coef in enumerate(coefs) if coef != 0]

#         if model_type == 'random_forest':
#             root_features = []
#             for estimator in best_model.estimators_:
#                 root_feature_index = estimator.tree_.feature[0]
#                 root_feature_name = X_train_selected.columns[root_feature_index]
#                 root_features.append(root_feature_name)
#         if model_type == 'gradient_boosting':
#             importances = best_model.feature_importances_
#             for i, feature in enumerate(selected_features):
#                 feature_coefficient_sum[feature] += importances[i]

#     return auc_scores


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.utils import resample

def auc_plot(yt, ypp):
    y_true = np.array(yt)
    y_scores = np.array(ypp)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(4, 3)) 
    lw = 2  # Line width
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label=f'ROC curve (AUC = {roc_auc:.2f})')
    # plt.fill_between(fpr, tpr_lower, tpr_upper, color='grey', alpha=0.3, label=f'{ci}% Confidence interval')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score
from sklearn.utils import resample

def compute_evaluation_matrix(y_true, y_scores, n_bootstraps=1000, ci=95):
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    original_cm = confusion_matrix(y_true, y_scores.round())
    original_accuracy = accuracy_score(y_true, y_scores.round())
    results = {
        "Confusion Matrix\n": original_cm,
        "Accuracy": original_accuracy,
        "Original AUC": roc_auc,
    }

    return results

# ============================================================================
# PANEL C: BOXPLOT - AUC Distribution by Group and Feature Set
# ============================================================================

import matplotlib.pyplot as plt
import seaborn as sns
from statannotations.Annotator import Annotator

def plot_panel_c_boxplot(combined_evaluation_df, save_path=None, figsize=(14, 6)):
    
    
    fig, ax = plt.subplots(figsize=figsize)

    # feature_colors = {
    #     'Baseline': '#7C9895',
    #     'Proteomics': '#DAA87C',
    #     'Metabolomics': '#92A5D1',
    #     'Overall': '#80C1C4'
    # }

    feature_colors = {
        'Basic': '#8ab07c',
        'Proteomics': '#f3a361',
        'Metabolomics': '#9b71aa',
        'Comprehensive': '#297270'
    }


    group_order = ['Full Cohort', 'Female', 'Male']
    feature_order = ['Basic', 'Proteomics', 'Metabolomics', 'Comprehensive']

    sns.boxplot(
        data=combined_evaluation_df,
        x='Groups',
        y='AUC',
        hue='Features',
        order=group_order,
        hue_order=feature_order,
        palette=feature_colors,
        ax=ax,
        width=0.65,
        gap=0.2,
        linewidth=2,
        dodge=True,
        boxprops=dict(edgecolor='black'),
        whiskerprops=dict(color='black', linewidth=1.8),
        capprops=dict(color='black', linewidth=1.8),
        medianprops=dict(color='black', linewidth=2.2),
        fliersize=4
    )

    # Define pairs for statistical comparison
    # Compare Baseline with each other feature within each group
    pairs = []
    for group in group_order:
        for feature in feature_order[1:]:  # Skip Basic itself
            pairs.append(((group, 'Basic'), (group, feature)))

    # Create annotator
    annotator = Annotator(
        ax, 
        pairs, 
        data=combined_evaluation_df,
        x='Groups', 
        y='AUC', 
        hue='Features',
        order=group_order,
        hue_order=feature_order
    )
    
    # Configure and apply annotations
    annotator.configure(
        test='Mann-Whitney',  # or 't-test_ind', 'Wilcoxon', etc.
        text_format='star',   # 'star' for */**, or 'simple' for p-values
        loc='inside',         # or 'outside'
        verbose=1,
        line_height=0.02,
        line_width=1.5,
        fontsize=10
    )
    
    annotator.apply_and_annotate()

    # 1. Remove the "Groups" label completely
    ax.set_xlabel('')

    # 2. Make "Combined", "Female", "Male" Larger and Bold
    ax.tick_params(axis='x', labelsize=14)
    plt.setp(ax.get_xticklabels(), fontweight='bold')

    # Y-axis styling
    ax.set_ylabel('AUC', fontsize=13, fontweight='bold')
    ax.tick_params(axis='y', labelsize=11)

    ax.set_title(
        'Panel C: AUC Distribution by Group and Feature Set',
        fontsize=14, fontweight='bold', pad=20
    )

    ax.set_ylim(-0.05, 1.5)  # Increased for annotations

    ax.legend(
        title=' ',
        loc='upper center',
        bbox_to_anchor=(0.5, -0.18),
        ncol=4,
        frameon=False,
        fontsize=14,
        title_fontsize=12
    )

    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')

    plt.show()
    return fig

def plot_panel_d_butterfly_enhanced(
    feature_selection_df_M, feature_selection_df_F, 
    demographic_cols, proteomic_cols, metabolomics_cols, 
    neuropsychological_cols, top_n=20, save_path=None,
    sort_by='total',
    label_font_size=12,
    figsize=(12, 6)
):
    
    # ... [Data Preparation Steps stay the same] ...
    # (Assuming the data prep code from your snippet is here)
    # ------------------------------------------------------
    # Create feature_type_map from column lists
    feature_type_map = {}
    for col in demographic_cols: feature_type_map[col] = 'Demographics'
    for col in proteomic_cols: feature_type_map[col] = 'Proteomics'
    for col in metabolomics_cols: feature_type_map[col] = 'Metabolomics'
    for col in neuropsychological_cols: feature_type_map[col] = 'Neuropsychological'
    
    male_top = feature_selection_df_M.head(top_n).copy()
    female_top = feature_selection_df_F.head(top_n).copy()
    all_features = list(set(male_top['Feature'].tolist() + female_top['Feature'].tolist()))
    
    plot_data = []
    for feature in all_features:
        male_count = male_top[male_top['Feature'] == feature]['Total Count'].values
        male_count = male_count[0] if len(male_count) > 0 else 0
        female_count = female_top[female_top['Feature'] == feature]['Total Count'].values
        female_count = female_count[0] if len(female_count) > 0 else 0
        feature_type = feature_type_map.get(feature, 'Other')
        
        plot_data.append({
            'Feature': feature, 'Male': male_count, 'Female': female_count,
            'Total': male_count + female_count, 'Difference': abs(male_count - female_count),
            'Type': feature_type
        })
    
    df_plot = pd.DataFrame(plot_data)
    
    # Sorting logic (Same as before)
    if sort_by == 'total': df_plot = df_plot.sort_values('Total', ascending=True)
    elif sort_by == 'male': df_plot = df_plot.sort_values('Male', ascending=True)
    elif sort_by == 'female': df_plot = df_plot.sort_values('Female', ascending=True)
    elif sort_by == 'difference': df_plot = df_plot.sort_values('Difference', ascending=True)
    elif sort_by == 'type':
        type_order = ['Demographics', 'Neuropsychological', 'Proteomics', 'Metabolomics', 'Other']
        df_plot['Type_Cat'] = pd.Categorical(df_plot['Type'], categories=type_order, ordered=True)
        df_plot = df_plot.sort_values(['Type_Cat', 'Total'], ascending=[True, True])
        df_plot = df_plot.drop('Type_Cat', axis=1)
    else: df_plot = df_plot.sort_values('Total', ascending=True)
    
    df_plot = df_plot.tail(top_n)
    
    # ------------------------------------------------------
    # PLOTTING STARTS HERE
    
    # Create figure with Female (ax1) on left, Male (ax2) on right
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=True)
    
    type_colors = {
        'Proteomics': '#f3a361',
        'Metabolomics': '#9b71aa',
        'Demographics': '#6A9ACE',          
        'Neuropsychological': '#1E803D',
        'Other': '#BEBADA'
    }
    
    y_pos = np.arange(len(df_plot))
    
    # --- FEMALE PLOT (LEFT) ---
    colors_female = [type_colors.get(t, '#BEBADA') for t in df_plot['Type']]
    ax1.barh(y_pos, df_plot['Female'], color=colors_female, alpha=0.85, 
            edgecolor='black', linewidth=0.7)
    
    ax1.set_xlabel('Selection Frequency (Female)', fontsize=13, fontweight='bold')
    ax1.set_title('Female', fontsize=15, fontweight='bold')
    ax1.invert_xaxis()  # Flips the bars to point left
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    
    # --- Y-AXIS LABELS (CONTROLLED HERE) ---
    # Since sharey=True, we set the labels on ax1 (the left axis)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(
        df_plot['Feature'], 
        fontsize=label_font_size,
        fontweight='bold'
    )
    
    # --- MALE PLOT (RIGHT) ---
    colors_male = [type_colors.get(t, '#BEBADA') for t in df_plot['Type']]
    ax2.barh(y_pos, df_plot['Male'], color=colors_male, alpha=0.85, 
            edgecolor='black', linewidth=0.7)
    
    ax2.set_xlabel('Selection Frequency (Male)', fontsize=13, fontweight='bold')
    ax2.set_title('Male', fontsize=15, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Remove y-ticks from the right plot to look cleaner
    ax2.tick_params(left=False) 

    # --- LEGEND & LAYOUT ---
    sort_label = {'total': 'Total', 'male': 'Male', 'female': 'Female', 
                  'difference': 'Diff', 'type': 'Type'}
    
    fig.suptitle(f'Panel D: Feature Selection Frequency by Sex (Sorted by {sort_label.get(sort_by, "Total")})', 
                fontsize=16, fontweight='bold', y=0.98)
    
    from matplotlib.patches import Patch
    present_types = df_plot['Type'].unique()
    legend_elements = [Patch(facecolor=type_colors[ftype], edgecolor='black', 
                            linewidth=0.7, label=ftype) 
                      for ftype in ['Proteomics', 'Metabolomics', 'Demographics', 
                                   'Neuropsychological', 'Other']
                      if ftype in present_types]
    
    fig.legend(handles=legend_elements, loc='lower center', ncol=len(legend_elements),
              fontsize=15, frameon=True, bbox_to_anchor=(0.5, 0.0),
              edgecolor='white', fancybox=False)
    
    # Adjust layout to make sure large labels fit
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
    
    plt.show()
    return fig



import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot_feature_coefficients_lollipop_enhanced(
    feature_coef_df, 
    demographic_cols=[],
    proteomic_cols=[],
    metabolomics_cols=[],
    neuropsychological_cols=[],
    top_n=20, 
    save_path=None,
    figsize=(12, 10), # Wider figure to accommodate text
    title='Feature Effects (Median & IQR)'
):
    # 1. Prepare Data
    plot_df = (
        feature_coef_df
        .nlargest(top_n, 'Abs_Mean_Coefficient')
        .sort_values('Median_Coefficient')
        .reset_index(drop=True)
    )

    # 2. Define Color Palette (Academic/Nature style)
    # You can change these hex codes to your preference
    # color_map = {
    #     'Demographics': '#404040',   # Dark Grey
    #     'Proteomics': '#E64B35',     # Red/Orange (Lancet style)
    #     'Metabolomics': '#00A087',   # Teal/Green
    #     'Neuropsychological': '#3C5488',     # Dark Blue
    #     'Other': '#B09C85'           # Neutral Beige/Brown
    # }
    color_map = {
        'Proteomics': '#f3a361',
        'Metabolomics': '#9b71aa',
        'Demographics': '#6A9ACE',          
        'Neuropsychological': '#1E803D',
        'Other': '#BEBADA'
    }
    
    
    # 3. Assign Colors to Rows based on feature name
    row_colors = []
    row_categories = []
    
    for feature in plot_df['Feature']:
        if feature in demographic_cols:
            cat = 'Demographics'
        elif feature in proteomic_cols:
            cat = 'Proteomics'
        elif feature in metabolomics_cols:
            cat = 'Metabolomics'
        elif feature in neuropsychological_cols:
            cat = 'Neuropsychological'
        else:
            cat = 'Other'
            
        row_categories.append(cat)
        row_colors.append(color_map[cat])

    y = np.arange(len(plot_df))
    fig, ax = plt.subplots(figsize=figsize)

    # 4. Draw IQR Lines (The "Stick")
    ax.hlines(
        y,
        plot_df['Q1_Coefficient'],
        plot_df['Q3_Coefficient'],
        color=row_colors,
        linewidth=4,
        alpha=0.6 # Slightly transparent for the range
    )

    # 5. Draw Median Points (The "Lollipop")
    ax.scatter(
        plot_df['Median_Coefficient'],
        y,
        color=row_colors,
        s=130,
        zorder=3,
        edgecolor='black', # Black edge makes them pop against grid
        linewidth=1.2
    )

    # 6. Zero Line
    ax.axvline(0, color='black', linewidth=1.5, linestyle='--', alpha=0.7)

    # 7. Axes & Labels
    ax.set_yticks(y)
    # ax.set_yticklabels(plot_df['Feature'], fontsize=10, fontfamily='sans-serif') # Smaller text
    ax.set_yticklabels(
        plot_df['Feature'], 
        fontsize=12,         # Increased from 10
        fontweight='bold',   # Added bold
        fontfamily='sans-serif'
    ) 
    ax.set_xlabel('Coefficient (Median with IQR)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    
    ax.set_xlabel('Coefficient (Median with IQR)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)

    # 8. Aesthetics (Spines and Grid)
    ax.grid(axis='x', linestyle=':', alpha=0.4, color='gray')
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines['left'].set_visible(False) # Clean look
    ax.spines['bottom'].set_linewidth(1.2)

    # 9. Smart Annotation (Prevent Overlap)
    # Calculate x-limits to ensure text fits
    x_min, x_max = ax.get_xlim()
    data_range = x_max - x_min
    padding = data_range * 0.04 # 2% padding
    
    for i, row in plot_df.iterrows():
        val = row['Median_Coefficient']
        
        # Decide if text goes left or right of the dot
        # If positive, put text to the right. If negative, to the left.
        # Add slight padding so it doesn't touch the dot
        if val > 0:
            ha = 'left'
            x_pos = val + padding
        else:
            ha = 'right'
            x_pos = val - padding

        ax.text(
            x_pos, i, 
            f"{val:.3f}",
            va='center',
            ha=ha,
            fontsize=11,
            color='#333333',
            # fontweight='bold'
            fontweight='medium'
        )

    # Expand x-axis limits slightly so text doesn't get cut off
    # We find the furthest point (either Q1/Q3 or the text position)
    all_values = pd.concat([plot_df['Q1_Coefficient'], plot_df['Q3_Coefficient']])
    min_val = all_values.min()
    max_val = all_values.max()
    margin = (max_val - min_val) * 0.15 # 15% margin
    ax.set_xlim(min_val - margin, max_val + margin)

    # 10. Dynamic Legend
    # Only create legend entries for categories that actually appear in the plot
    unique_cats = sorted(list(set(row_categories)))
    legend_patches = [
        mpatches.Patch(color=color_map[cat], label=cat) 
        for cat in unique_cats
    ]
    
    ax.legend(
        handles=legend_patches, 
        loc='lower right', 
        frameon=True, 
        fontsize=14,
        # edgecolor='white',
        edgecolor='black',
        facecolor='white',
        framealpha=1
    )

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
        
    plt.show()


def _bh_qvalues(pvals):
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    q = np.empty(m)
    q_sorted = p[order] * m / np.arange(1, m + 1)
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q[order] = np.clip(q_sorted, 0, 1)
    return q

def filter_significant_features(coef_df, alpha=0.05, use_t=True,
                                fdr=False, min_count=None):
    """
    Wald/t-test on aggregated lasso coefficients; filter beta != 0.
    Needs columns: Feature, Mean_Coefficient, SE_Coefficient, Count,
    Abs_Mean_Coefficient. Returns (significant_df, feature_list).
    """
    df = coef_df.copy()
    se = df["SE_Coefficient"].replace(0, np.nan)
    stat = df["Mean_Coefficient"] / se

    if use_t:
        dfree = (df["Count"] - 1).clip(lower=1)
        df["p_value"] = 2 * stats.t.sf(np.abs(stat), dfree)
    else:
        df["p_value"] = 2 * stats.norm.sf(np.abs(stat))
    df["p_value"] = df["p_value"].fillna(1.0)   # undefined test -> not significant

    sig_col = "p_value"
    if fdr:
        df["q_value"] = _bh_qvalues(df["p_value"].values)
        sig_col = "q_value"

    keep = df[sig_col] < alpha
    if min_count is not None:
        keep &= df["Count"] >= min_count

    sig = (df[keep]
           .sort_values("Abs_Mean_Coefficient", ascending=False)
           .reset_index(drop=True))
    return sig, sig["Feature"].tolist()


def filter_stable_features(feature_selection_df, n_evaluations,
                           min_frequency=0.5):
    """Keep features selected in a minimum fraction of outer CV fits."""
    if n_evaluations <= 0:
        raise ValueError("n_evaluations must be positive")
    if not 0 < min_frequency <= 1:
        raise ValueError("min_frequency must be in (0, 1]")

    stable = feature_selection_df.copy()
    stable["Selection Frequency"] = stable["Total Count"] / n_evaluations
    stable = stable[stable["Selection Frequency"] >= min_frequency]
    stable = stable.sort_values(
        ["Selection Frequency", "Total Count"], ascending=False
    ).reset_index(drop=True)
    return stable, stable["Feature"].tolist()
