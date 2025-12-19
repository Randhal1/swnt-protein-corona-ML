# Import necessary libraries

# Basic libraries
import pandas as pd

# ML libraries
from sklearn.preprocessing import MinMaxScaler as scaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import metrics
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
from imblearn.over_sampling import SMOTE

class ORFC:
    def __init__(self, training_data, target_feature='Corona', random_state=2005, k=91, predictor_features=None):

        # Define what is this model supposed to do
        if predictor_features is None:
            predictor_features = list(training_data.columns)[2:]

        scale = scaler().fit(training_data[predictor_features])

        scaled_predictors = pd.DataFrame(scale.transform(training_data[predictor_features]), columns=predictor_features)

        k_fold_splits = 100
        predictions = pd.DataFrame()

        # Select the top k features
        self.selector = SelectKBest(f_classif, k=k)
        X_new = self.selector.fit_transform(scaled_predictors, training_data["Protein names"]) #X_train_set.values #

        predictor_features = [i for i in predictor_features if self.selector.get_support()[predictor_features.index(i)]]
        self.predictor_features = predictor_features

        scale = scaler().fit(training_data[predictor_features])
        self.scale = scale

        scaled_predictors = pd.DataFrame(scale.transform(training_data[predictor_features]), columns=predictor_features)
        
        print("Selected Features:", predictor_features, X_new.shape)

        df_local_features_train = pd.DataFrame(X_new.copy())#df_local_features_train.copy() #pd.DataFrame(X_new.copy()) #scaled_df.copy()

        feature_imp = pd.DataFrame(columns=list(df_local_features_train.columns))
        first_frame = True
        correctness_frame = pd.DataFrame()
        metrics_frame = pd.DataFrame()
        self.predictor_features = predictor_features

        ## split up our data
        i = 0

        sss = StratifiedShuffleSplit(n_splits=k_fold_splits, test_size=0.1, random_state=random_state)

        for train_index, test_index in sss.split(df_local_features_train, training_data[target_feature]): # comment our if doing cross fluid
            X_train = df_local_features_train.iloc[train_index] # remove subsetting for cross fluid tests
            X_test = df_local_features_train.iloc[test_index] # change dataframe for cross fluid tests
            y_train = training_data[target_feature].iloc[train_index] # remove subsetting for cross fluid tests
            y_test = training_data[target_feature].iloc[test_index] # change dataframe for cross fluid tests
        
            # Create and Train
            rfc=RandomForestClassifier(criterion='entropy', min_impurity_decrease = 0.02,  min_samples_split=2, max_depth = 10, max_features = 'sqrt',
            n_jobs=-1, ccp_alpha=0.01, random_state=random_state, n_estimators=700) 
        
            
            #sme = SMOTE(random_state=random_state, sampling_strategy=0.7, n_jobs=-1, k_neighbors=12)
            sme = SMOTE(random_state=random_state, sampling_strategy=0.7, k_neighbors=12)

            X_train_oversampled, y_train_oversampled = sme.fit_resample(X_train, y_train)
            # X_train_oversampled, y_train_oversampled = X_train, y_train # can be used to pass smote if needed for an experiment
            rfc.fit(X_train_oversampled,y_train_oversampled)


            if first_frame:  # Initialize 
                first_frame = False  # Don't Come back Here
                
                datadict = {'true':y_test.to_numpy(), 'estimate':rfc.predict(X_test), 'probability':rfc.predict_proba(X_test)[:, 1]}
                
                correctness_frame = pd.DataFrame(data=datadict)
                correctness_frame['round'] = i

                metrics_dict = {'AUC':metrics.roc_auc_score(y_test, rfc.predict_proba(X_test)[:, 1]),
                'Accuracy':rfc.score(X_test, y_test), 'Recall':recall_score(y_test, rfc.predict(X_test)), 
                'Precision':precision_score(y_test, rfc.predict(X_test)), 'F1':f1_score(y_test, rfc.predict(X_test))}
                
                metrics_frame = pd.DataFrame.from_dict(data=metrics_dict,orient='index').transpose()
                metrics_frame['Round'] = i

                # can be used if you want to track prediction during shuffle split - saves in another cell
                predictions = pd.DataFrame()
                predictions['Protein Name'] = training_data['Protein names']
                predictions['In Corona Probability'] = rfc.predict_proba(scaled_predictors)[:, 1]
                predictions['Round'] = i
                predictions['Test Accuracy'] = metrics_dict['Accuracy']
                predictions['Test Recall'] = metrics_dict['Recall']
                predictions['Test Precision'] = metrics_dict['Precision']
                predictions['Test AUC'] = metrics_dict['AUC']

                
            else:
                datadict = {'true':y_test.to_numpy(), 'estimate':rfc.predict(X_test), 'probability':rfc.predict_proba(X_test)[:, 1]}
                revolve_frame = pd.DataFrame(data=datadict)
                revolve_frame['round'] = i
                #correctness_frame = correctness_frame.append(revolve_frame, ignore_index=True)
                correctness_frame = pd.concat([correctness_frame, revolve_frame], ignore_index=True)

                metrics_dict = {'AUC':metrics.roc_auc_score(y_test, rfc.predict_proba(X_test)[:, 1]),
                'Accuracy':rfc.score(X_test, y_test), 'Recall':recall_score(y_test, rfc.predict(X_test)), 
                'Precision':precision_score(y_test, rfc.predict(X_test)), 'F1':f1_score(y_test, rfc.predict(X_test))}
                metrics_revolve_frame = pd.DataFrame.from_dict(data=metrics_dict, orient='index').transpose()
                metrics_revolve_frame['Round'] = i
                #metrics_frame = metrics_frame.append(metrics_revolve_frame, ignore_index=True)
                metrics_frame = pd.concat([metrics_frame, metrics_revolve_frame], ignore_index=True)

                # can be used if you want to track prediction during shuffle split - saves in another cell
                pred_rev = pd.DataFrame()
                pred_rev['Protein Name'] = training_data['Protein names']
                pred_rev['In Corona Probability'] = rfc.predict_proba(scaled_predictors)[:, 1]
                pred_rev['Round'] = i
                pred_rev['Test Accuracy'] = metrics_dict['Accuracy']
                pred_rev['Test Recall'] = metrics_dict['Recall']
                pred_rev['Test Precision'] = metrics_dict['Precision']
                pred_rev['Test AUC'] = metrics_dict['AUC']

                #predictions = predictions.append(pred_rev, ignore_index=True)
                predictions = pd.concat([predictions, pred_rev], ignore_index=True)
        
            feature_imp.loc[i] = pd.Series(rfc.feature_importances_,index=list(df_local_features_train.columns))
            
            i += 1
        
        self.metrics_frame = metrics_frame
        self.rfc = rfc

    def rmetrics(self):
        return self.metrics_frame.mean()
    
    def mkpred(self, X_new_data):
        y_new_data = X_new_data['Corona']
        #X_new_data = X_new_data.drop(['Protein names', 'mass', 'Corona'], axis=1)
        X_new_data = pd.DataFrame(self.scale.transform(X_new_data[self.predictor_features]), columns=self.predictor_features)

        # Assuming you have the following dataframes for the new prediction:
        # X_new_data: DataFrame of features (must be the 91 selected features)
        # y_new_data: Series/Array of true labels

        # --- 1. Make Predictions using the Last Trained Model (rfc) ---
        # Get predicted class (0 or 1)
        y_pred_new = self.rfc.predict(X_new_data)

        # Get prediction probabilities for the positive class (used for AUC)
        y_proba_new = self.rfc.predict_proba(X_new_data)[:, 1]

        # Area Under the ROC Curve
        new_auc = roc_auc_score(y_new_data, y_proba_new)

        # Recall (Sensitivity)
        new_recall = recall_score(y_new_data, y_pred_new)

        # Precision (Positive Predictive Value)
        new_precision = precision_score(y_new_data, y_pred_new)

        # F1-Score (Harmonic mean of Precision and Recall)
        new_f1 = f1_score(y_new_data, y_pred_new)

        # You can also get the overall Accuracy
        new_accuracy = self.rfc.score(X_new_data, y_new_data)

        # --- 3. Display Results ---
        print(f"Metrics on New Data:")
        print(f"  AUC:       {new_auc:.4f}")
        print(f"  Recall:    {new_recall:.4f}")
        print(f"  Precision: {new_precision:.4f}")
        print(f"  F1-Score:  {new_f1:.4f}")
        print(f"  Accuracy:  {new_accuracy:.4f}")

        # --- 4. Visualize the Confusion Matrix (Optional but Recommended) ---
        # This shows the breakdown of correct and incorrect predictions.
        conf_matrix = metrics.confusion_matrix(y_new_data, y_pred_new)
        print("\nConfusion Matrix:")
        print(conf_matrix)