
# Introductionzz

This report evaluates the performance of different regression models at predicting the quality of a variety of Portuguese wines. A dataset has been used which describes red wine via 11 different features, including alcohol content, fixed acidity and quality. The task has been to assess the efficacy of different machine learning models in predicting the quality of a wine given the other features.  

This reports discusses four regression models for that were created to predict wine quality. This includes 3 linear basis function models polynomial, bayesian and radial basis as well as K-Nn. These models were selected to sample a selection of some of the most popular Linear regression models. By optimising each model, a fair comparison of  each models efficacy was made using the same scoring for each.

## Performance Evaluation

The metrics with which we have evaluated the performance of these models are as follows:

- Root mean square error (${E}_{RMS}$): this is measure of prediction accuracy. RMSE disproportionately affects the points further from the actual results, therefore favouring predictions with small variance. This ensures that the metric is more sensitive to outlier values. This is the primary metric that we have used to asses our models. (${E}_{RMS}$) is described by:

\begin{align}{ E }_{ RMS }=\sqrt { \frac{ 2E({ { \mathbf{w}} }^{ * })}{ N }}
\end{align}

- Mean absolute error (${ E }_{ M }$): this is a measure of the actual distance of the error from the prediction. This is less sensitive to large errors than   ${E}_{RMS}$, but gives a clear measure of how far predictions on average deviate from the target.
- Mean absolute percentage error (${ E }_{ M \% }$): presenting mean absolute error as a percentage can help gauge the performance of the model in a more conceptual  way, helping to gauge how big the error is
- Median absolute error (${ E }_{ \tilde { x }  }$): This is the middle most error of predictions. This measure is useful as it is unaffected by outliers giving a picture of how the general accuracy of the model
- Variance (${ \sigma }^{ 2 }$): this measure highlights the precision marked by the spread of predictions. A low variance indicates that prediction are all made in a similar region with fewer outliers. However a model could have some issues with it's accuracy, this is why the variance needs to be used in conjunction with a mean value.  

A function named `error_score.py`, was created which analysed predictions, making sure all the models were evaluated equally; enabling direct comparisons to be made.

## Cross Validation

To maximise the data that we have at our disposal we validate our models using K-fold cross validation. This separates the data into K separate folds, of which K-1 are training folds and one is the testing fold. The model is then trained and tested K times, each time using a different fold for testing. We then take the average error values of all the folds. This ensures we reduce our bias (more fitting data) and variance (more validation data) as all of the data becomes both training and testing data.

\begin{table}[h]
\centering
\caption{Example showing data splitting for 5-Fold cross validation}
\label{tab:validation}
\begin{tabular}{ccccccc}
\cline{1-6}
\multicolumn{1}{l}{\cellcolor[HTML]{00171F}{\color[HTML]{FFFFFF} }} & \cellcolor[HTML]{00171F}{\color[HTML]{FFFFFF} fold 1} & \cellcolor[HTML]{00171F}{\color[HTML]{FFFFFF} fold 2} & \cellcolor[HTML]{00171F}{\color[HTML]{FFFFFF} fold 3} & \cellcolor[HTML]{00171F}{\color[HTML]{FFFFFF} fold 4} & \cellcolor[HTML]{00171F}{\color[HTML]{FFFFFF} fold 5} & \multicolumn{1}{l}{} \\ \cline{1-6}
\multicolumn{1}{|c|}{\cellcolor[HTML]{00171F}{\color[HTML]{FFFFFF} iteration 1}} & \multicolumn{1}{c|}{\cellcolor[HTML]{FFCCC9}test} & \multicolumn{1}{c|}{\cellcolor[HTML]{9AFF99}train} & \multicolumn{1}{c|}{\cellcolor[HTML]{9AFF99}train} & \multicolumn{1}{c|}{\cellcolor[HTML]{9AFF99}train} & \multicolumn{1}{c|}{\cellcolor[HTML]{9AFF99}train} &  \\ \cline{1-6}
\multicolumn{1}{|c|}{\cellcolor[HTML]{00171F}{\color[HTML]{FFFFFF} iteration 2}} & \multicolumn{1}{c|}{\cellcolor[HTML]{9AFF99}train} & \multicolumn{1}{c|}{\cellcolor[HTML]{FFCCC9}test} & \multicolumn{1}{c|}{\cellcolor[HTML]{9AFF99}train} & \multicolumn{1}{c|}{\cellcolor[HTML]{9AFF99}train} & \multicolumn{1}{c|}{\cellcolor[HTML]{9AFF99}train} &  \\ \cline{1-6}
\multicolumn{1}{|c|}{\cellcolor[HTML]{00171F}{\color[HTML]{FFFFFF} iteration 3}} & \multicolumn{1}{c|}{\cellcolor[HTML]{9AFF99}train} & \multicolumn{1}{c|}{\cellcolor[HTML]{9AFF99}train} & \multicolumn{1}{c|}{\cellcolor[HTML]{FFCCC9}test} & \multicolumn{1}{c|}{\cellcolor[HTML]{9AFF99}train} & \multicolumn{1}{c|}{\cellcolor[HTML]{9AFF99}train} &  \\ \cline{1-6}
\multicolumn{1}{|c|}{\cellcolor[HTML]{00171F}{\color[HTML]{FFFFFF} iteration 4}} & \multicolumn{1}{c|}{\cellcolor[HTML]{9AFF99}train} & \multicolumn{1}{c|}{\cellcolor[HTML]{9AFF99}train} & \multicolumn{1}{c|}{\cellcolor[HTML]{9AFF99}train} & \multicolumn{1}{c|}{\cellcolor[HTML]{FFCCC9}test} & \multicolumn{1}{c|}{\cellcolor[HTML]{9AFF99}train} &  \\ \cline{1-6}
\multicolumn{1}{|c|}{\cellcolor[HTML]{00171F}{\color[HTML]{FFFFFF} iteration 5}} & \multicolumn{1}{c|}{\cellcolor[HTML]{9AFF99}train} & \multicolumn{1}{c|}{\cellcolor[HTML]{9AFF99}train} & \multicolumn{1}{c|}{\cellcolor[HTML]{9AFF99}train} & \multicolumn{1}{c|}{\cellcolor[HTML]{9AFF99}train} & \multicolumn{1}{c|}{\cellcolor[HTML]{FFCCC9}test} &  \\ \cline{1-6}
\multicolumn{1}{l}{} & \multicolumn{1}{l}{} & \multicolumn{1}{l}{} & \multicolumn{1}{l}{} & \multicolumn{1}{l}{} & \multicolumn{1}{l}{} & \multicolumn{1}{l}{} \\ \cline{1-1} \cline{7-7}
\multicolumn{1}{|l|}{\cellcolor[HTML]{333333}{\color[HTML]{FFFFFF} validation phase}} & \multicolumn{1}{l}{} & \multicolumn{1}{l}{} & \multicolumn{1}{l}{} & \multicolumn{1}{l}{} & \multicolumn{1}{l|}{} & \multicolumn{1}{l|}{\cellcolor[HTML]{68CBD0}validation} \\ \cline{1-1} \cline{7-7}
\end{tabular}
\end{table}

In order to validate our model on previously unseen test data we held out a validation data set. This set is 10% of the original data. This means we perform K-fold cross validation on the other 90% of the data.

