# Copyright (C) 2024  Adam Jones  All Rights Reserved
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
# 
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


# libraries needed
library(readxl)
library(pimeta)
library(DescTools)

# constants to use
seed = 12
num_boot = 10001
alpha_for_CIs = 0.05
alpha_for_tests = alpha_for_CIs/2  # should be half for non-inferiority tests
alternative = "greater"


####
######## calculate the meta-analysis results

meta_input <- read_excel("meta_input.xlsx")

# remove the excluded
meta_input <- subset(meta_input, !meta_input$exclude)

# fill se using std and n
fill_se = is.na(meta_input$se) & !is.na(meta_input$std)
meta_input[fill_se, c("se")] = meta_input[fill_se, c("std")]/sqrt(meta_input[fill_se, c("n")])

# fill se using CIs
# all provided intervals are 95%
alpha_for_CIs_meta_studies = 0.05
# could use t-distribution, since n is small
# but norm is probably the one that was used (not specified)
# df_list = unname(unlist(meta_input[fill_se, c("n")])) - 1
# crit_for_meta_studies = 2*qt(1-alpha_for_CIs_meta_studies/2, df=df_list) # t distribution
crit_for_meta_studies = 2*qnorm(1-alpha_for_CIs_meta_studies/2)  # norm distribution
fill_se = is.na(meta_input$se) & !is.na(meta_input$lower_ci)
meta_input[fill_se, c("se")] = (meta_input[fill_se, c("upper_ci")] - meta_input[fill_se, c("lower_ci")])/crit_for_meta_studies

# fill in std using se (not necessary, but possible)
fill_std = !is.na(meta_input$se) & is.na(meta_input$std)
meta_input[fill_std, c("std")] = meta_input[fill_std, c("se")] * sqrt(meta_input[fill_std, c("n")])

# calculate the mu, CIs, PIs
stages = c("All", "Wake", "N1", "N2", "N3", "REM")
results <- data.frame(matrix(ncol = 12, nrow = 6))
colnames(results) <- c('stage', 'n', 'tau2h', 'i2h', 'bias_p', 'bias_mean', 'mean', 'sd', 'lci', 'uci', 'lpi', 'upi')

print("calculate meta-analysis results")
for (i in 1:6) {
    stage = stages[i]
    print(stage)
    kappas = unlist(meta_input[meta_input$stage == stage, c("kappa")])
    ses = unlist(meta_input[meta_input$stage == stage, c("se")])
    
    # get the main results using the default DerSimonian & Laird method
    cima_result = pimeta::cima(y=kappas, se=ses, method="DL", alpha=alpha_for_CIs)
    bias_result = meta::metabias(cima_result$y, seTE=cima_result$se, method.bias = "Thompson")
    results[i, c("stage")] <- stage
    results[i, c("n")] <- length(kappas)
    results[i, c("tau2h")] <- cima_result$tau2h
    results[i, c("i2h")] <- cima_result$i2h
    results[i, c("mean")] <- cima_result$muhat 
    results[i, c("sd")] <- sqrt(cima_result$vmuhat)
    results[i, c("lci")] <- cima_result$lci
    results[i, c("uci")] <- cima_result$uci
    results[i, c("bias_p")] <- bias_result$p.value
    results[i, c("bias_mean")] <- bias_result$intercept
    
    # get the prediction intervals using the bootstrap method
    pima_result = pimeta::pima(y=kappas, se=ses, method="boot", B=num_boot, seed=seed, alpha=alpha_for_CIs)
    results[i, c("lpi")] <- pima_result$lpi
    results[i, c("upi")] <- pima_result$upi
}


####
######## get the testing set results

# get model data
main_model_data <- read_excel("../2_intermediate_data/main_model_data.xlsx", guess_max=4000)

# take only the testing set
main_model_data <- subset(main_model_data, main_model_data$set == 3)

# rename two columns
colnames(main_model_data)[colnames(main_model_data) == 'o_kappa'] = 'all_kappa'
colnames(main_model_data)[colnames(main_model_data) == 'w_kappa'] = 'wake_kappa'


# studies
row_count = dim(results)[1]
studies = c("all", "ccshs", "cfs", "chat", "mesa", "wsc")


column_suffix = c("_p", "_mean", "_lci", "_uci")
for (j in 1:length(column_suffix)) {
    for (i in 1:length(studies)) {
        # add a new column
        results$new_column1 <- rep(NA, row_count)
        column_name = paste0(studies[i], column_suffix[j] , sep ="")
        names(results)[(dim(results)[2]):(dim(results)[2])] = c(column_name)
    }
}


for (i in 1:6) {
    stage = stages[i]
    print(stage)
    column_name = paste0(tolower(stage), "_kappa", "")
    for (j in 1:6) {
        study = studies[j]
        print(study)
        if (j == 1) {
            kappas = unname(unlist(main_model_data[,c(column_name)]))
        } else {
            kappas = unname(unlist(main_model_data[main_model_data$study==study, c(column_name)]))
        }
        
        test_result = t.test(kappas, alternative=alternative, conf.level=(1-alpha_for_tests), mu=results[results$stage==stage, c("lci")])
        
        # save the result
        temp_column_name = paste0(study, "_p", "")
        results[i, c(temp_column_name)] = test_result$p.value
        
        temp_column_name = paste0(study, "_mean", "")
        results[i, c(temp_column_name)] = mean(kappas)
        
        # confidence intervals assuming t distribution
        # (t, because it is a sample with small n)
        crit = qt(1-alpha_for_CIs/2, df=(length(kappas)-1))
        temp_column_name = paste0(study, "_lci", "")
        results[i, c(temp_column_name)] = mean(kappas)-crit*(sd(kappas)/sqrt(length(kappas)))
        temp_column_name = paste0(study, "_uci", "")
        results[i, c(temp_column_name)] = mean(kappas)+crit*(sd(kappas)/sqrt(length(kappas)))
    }
}


####
######## get the real-time testing set results

study = "rt_all"
for (j in 1:length(column_suffix)) {
    # add a new column
    results$new_column1 <- rep(NA, row_count)
    column_name = paste0(study, column_suffix[j] , sep ="")
    names(results)[(dim(results)[2]):(dim(results)[2])] = c(column_name)
}


rt_model_data <- read_excel("../2_intermediate_data/real_time_model.xlsx", guess_max=4000)
# rename columns
colnames(rt_model_data)[colnames(rt_model_data) == 'All'] = 'all_kappa'
colnames(rt_model_data)[colnames(rt_model_data) == 'Wake'] = 'wake_kappa'
colnames(rt_model_data)[colnames(rt_model_data) == 'N1'] = 'n1_kappa'
colnames(rt_model_data)[colnames(rt_model_data) == 'N2'] = 'n2_kappa'
colnames(rt_model_data)[colnames(rt_model_data) == 'N3'] = 'n3_kappa'
colnames(rt_model_data)[colnames(rt_model_data) == 'REM'] = 'rem_kappa'


for (i in 1:6) {
    stage = stages[i]
    print(stage)
    column_name = paste0(tolower(stage), "_kappa", "")
    for (j in 1:1) {
        print(study)
        if (j == 1) {
            kappas = unname(unlist(rt_model_data[,c(column_name)]))
        } else {
            kappas = c()
        }
        
        # using t test against mu = lci
        test_result = t.test(kappas, alternative=alternative, conf.level=(1-alpha_for_tests), mu=results[results$stage==stage, c("lci")])

        # save the result
        temp_column_name = paste0(study, "_p", "")
        results[i, c(temp_column_name)] = test_result$p.value
        
        temp_column_name = paste0(study, "_mean", "")
        results[i, c(temp_column_name)] = mean(kappas)
        
        # confidence intervals assuming t distribution
        # (t, because it is a sample with small n)
        crit = qt(1-alpha_for_CIs/2, df=(length(kappas)-1))
        temp_column_name = paste0(study, "_lci", "")
        results[i, c(temp_column_name)] = mean(kappas)-crit*(sd(kappas)/sqrt(length(kappas)))
        temp_column_name = paste0(study, "_uci", "")
        results[i, c(temp_column_name)] = mean(kappas)+crit*(sd(kappas)/sqrt(length(kappas)))
    }
}


#### adjust p-values using Hochberg's procedure
p_list = c(studies, "rt_all")

# create the additional columns for the adjusted p-value
for (j in 1:length(p_list)) {
    # add a new column
    results$new_column1 <- rep(NA, row_count)
    study = p_list[j]
    column_name = paste0(study, "_p_adj" , sep ="")
    names(results)[(dim(results)[2]):(dim(results)[2])] = c(column_name)
}

# go one stage at a time, since that is the multiple comparison
for (i in 1:6) {
    stage = stages[i]
    print(stage)
    
    # get the list of p-values
    p_values = c()
    for (j in 1:length(p_list)) {
        study = p_list[j]
        temp_column_name = paste0(study, "_p", "")
        p_temp = results[i, c(temp_column_name)]
        p_values = c(p_values, p_temp)
    }
    
    # adjust the p-values for multiple comparisons
    p_values_adj = p.adjust(p_values, method="hochberg")
    
    # now, store those adjusted p-values
    for (j in 1:length(p_list)) {
        study = p_list[j]
        temp_column_name = paste0(study, "_p_adj", "")
        results[i, c(temp_column_name)] = p_values_adj[j]
    }
    
}




#### save results
write.csv(results, file="meta_results_lci.csv", row.names=FALSE)

