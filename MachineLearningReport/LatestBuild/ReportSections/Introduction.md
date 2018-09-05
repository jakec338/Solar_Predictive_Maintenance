
# Introduction

## Problem
blah blahAs the world moves towards a decarbonised economy, the share of power production going to solar has increased dramatically. This increase in many countries has been fuelled by subsidies governments interested in achieving the sustainability targets that have been set both by themselves and the wider international community (ref, Bohringer paper). However in so doing so capacity has been expanded at a time when typically energy demand is falling (ref, eurostat). In many countries this has led to a slump in wholesale electricity prices (ref, Bohringer paper). For large scale utilities this has meant that profit margins have become increasingly thin. 

As a result, solar utilities are paying more attention to what had previously been considered marginal losses. Three days of downtime results in a loss equivalent to 1% yield loss for solar plants, seriously impacting profitability in the current climate (ref, egen.io). If this downtime can be reduced, this will have a significant impact on the profitability of plant. If maintenance issues can be predicted in advance this can decrease the downtime of a panel or, ideally, prevent any downtime being experience at all. 

In a report analysing the potential impact of machine learning on several industries, predictive maintenance was found to have the greatest impact on the energy sector (ref, mckinsey). Solar panel plants contain a variety of different sensors. A typical 5MW solar PV plant has over 800 sensors which generate about 1 Terabyte of data in a week (ref, algoengines). This quantity of data makes it difficult to analyse in-house by plant operators but also provides a rich data set on which to use machine learning techniques to predict maintenance issues. 


## Aims and Goals
The prediction of maintenance issues is the primary concern for the client. The data has been provided in 2 data sets. The first is the SCADA (Supervisory Control and Data Access) data, the second containing the error codes that have been produced by each of the 6 inverters, segmented into 10 modules each. The error codes and their meanings are supplied by the Solar Panel manufacturer, Power Electronics (appendix). By analysing both, the project aim is to explore the potential for machine learning techniques to predict the generation of error codes. This can be achieved either as a binary class (will this inverter produce error code 19 in the next two days for example), or can give a probabilistic measurement of the likelihood of an error code being produced within a certain time frame. 

Should time permit, I will begin to put the model into production with renewable.ai?s engineering team. This may involve analysing a regular stream of data from the plant and producing an intelligible product for the end user. 

My personal aims are to deepen my knowledge or R and Python languages, both of which are likely to be used over the duration of this project. As well as this, I hope to gain a better understanding of the work flow of a typical data science project and of the wider industry. 

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

