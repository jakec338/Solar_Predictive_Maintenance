# kNN Regression

\begin{table}[h]
\centering
\caption{kNN Regression Error Scores}
\label{tab:polyerror}
\begin{tabular}{|l|l|l|l|l|}
\hline
\rowcolor[HTML]{00171F}
\multicolumn{1}{|c|}{\cellcolor[HTML]{00171F}{\color[HTML]{FFFFFF} ${E}_{RMS}$}} & \multicolumn{1}{c|}{\cellcolor[HTML]{00171F}{\color[HTML]{FFFFFF} ${E}_{M}$}} & \multicolumn{1}{c|}{\cellcolor[HTML]{00171F}{\color[HTML]{FFFFFF} ${E}_{\tilde{x}}$}} & \multicolumn{1}{c|}{\cellcolor[HTML]{00171F}{\color[HTML]{FFFFFF} ${E}_{MP}$}} & \multicolumn{1}{c|}{\cellcolor[HTML]{00171F}{\color[HTML]{FFFFFF} ${\sigma}^{2}$}} \\ \hline
0.7933                                                               & 0.6289                                                               & 0.5                                                             & 11.1712                                                           & 0.6280                                                           \\ \hline
\end{tabular}
\end{table}

## Overview

The k-nearest neighbors regression splits data into target and training data sets. It predicts the target set by finding the average of the k-nearest values in the training set of data. It is therefore considered a 'lazy' algorithm as it does not feature a training phase and therefore doesn't generate a function which can later be tested. 

\begin{align}
	\hat { y } \quad =\quad f({ x })\quad =\quad \frac { 1 }{ k } \sum _{ { x }_{ n }\in { ℕ }_{ k }(x) }^{ k }{ { t }_{ n } } 
	\label{knn_func}
\end{align}

The distance between two inputs is calculated using the Minkowski distance function. This function takes a parameter **p** which determines whether the function takes the Euclidian (p = 1) or the Manhattan distance (p = 2) between the input rows. 

\begin{align}
	D({ x }_{ i },{ x }_{ j })\quad =\quad { \left( \sum _{ l=1 }^{ d }{ { \left| { x }_{ il }-{ x }_{ jl } \right|  }^{ 1/p } }  \right)  }^{ p }
	\label{distance_func}
\end{align}



## Method

- Select number of folds for cross-validation and number of **k** values to test
- Create cross-validation folds given size of data **N**
- A dictionary of matrix of errors is created to hold the different error terms for each fold iteration
- For each fold separate the data into training and target sets
- Plug the training and target data sets into the **kNN regression function**
- **If k = 1:** For each row in the target set find the Euclidian/ Manhattan distance between it and each row of the training set. Store these distances in the columns of the 2-dimensional distances matrix. Create a matrix with corresponding training targets for each training row. Sort these two arrays based on increasing distance and store in the sort matrix
- Now there is a matrix that holds all the distances for all targets
- For each value of k take the first k values of the target column of the sort matrix and divide their sum by **k** 
- This is the kNN predicted target value which stored in the kNN_targets array
- Given the actual target values and kNN values compute the error values (RMSE, Mean, Median etc.) and store them in the error matrix
- Repeat this for all values of **k**
- Return the error matrix for that iteration of cross-validation
- Repeat until there are *K* number of error matrices for the *K* number of folds
- Take the average of the *K* matrices to get an error matrix with the average errors
- Find the k value for the minimum error of each error type	


## Results

\begin{figure}[H]
\centering
\begin{minipage}{.49\textwidth}
  \centering
  \includegraphics[trim = 0 0 0 0, clip, width=1\textwidth]{RMSvskNN.pdf}
 \caption{$E_{RMS}$ Variation with k}
 \label{fig:rmskNN}
\end{minipage}
\hfill
\begin{minipage}{.49\textwidth}
  \centering
   \includegraphics[trim = 0 0 0 0, clip, width=1\textwidth]{VariancevskNN.pdf}
   \caption{${\sigma}^{2}$ Variation with k}
  \label{fig:varkNN}
\end{minipage}
\vspace{-20pt}
\end{figure}

## Discussion

As seen on Figure \ref{fig:rmskNN} the kNN regression root mean square error is minimized at a k value of **18** using the Manhattan distance measure. This means that the method returns the most accurate results when distances are compared with the 18 nearest neighbours (Note: The k-value varies as tests are run due to the random nature of our cross-validation method). Lower values of **k** result in gross over-fitting as seen by the sharp increase in the error value.  

The kNN by nature has high-variance (\ref{fig:varkNN}) but that can be accounted for by increasing the number of neighbours affecting the prediction (**k**) as seen on the second graph above.

The model is simple to implement and gives immediate results but unfortunately is very resource intensive. Each time kNN is run it needs to iterate through each training row for every target then sort the data, which makes it both computionally and storage intensive (especially with large training data sets). Another major disadvantage is that it is a so-called 'lazy learner' as it does not return any generalized model which could be used for further testing.

