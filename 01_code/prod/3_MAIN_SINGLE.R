#-- Train a single model on all the features (No feature selection)




rm(list=ls())

set.seed(123)


library(data.table)
library(lubridate)
library(mlr)
library(ggplot2)
library(xgboost)
library(crayon)
library(plotly)

#--- Directories
data_output_dir<-"02_data/output/"
data_input_dir<-"02_data/input/"
data_intermediate_dir<-"02_data/intermediate/"
models_dir<-"03_models/"


#------------------------------------------------------------#
##############################################################
#------------------------------------------------------------#


#------------------------------------------------------------#
################## DEFINE THE CONFIGURATIONS #################
#------------------------------------------------------------#

config_file <- data.table(
  SL = 15, # Stop loss
  PF = 1,  # Profit factor
  test_portion = 0.3, # Out of sample test part for final evaluation
  FEAT_SEL_get_feature_importance = F, # Flag for getting the feature importance
  featSelProb=0.8,
  FEAT_SEL_nrounds = 10, # Number of trees used in feature selection model
  FEAT_SEL_eta = 0.1,    # Learning rate of the xgboost in feature selection
  FEAT_SEL_initial.window = 1e4, # Portion of the dataset to be used for training
  FEAT_SEL_horizon = 1e3,        # Future samples to be used for testing
  FEAT_SEL_window_type = "FixedWindowCV",# "GrowingWindowCV", # "FixedWindowCV",
  FEAT_SEL_max_iterations = 2e2,
  TRAIN_nrounds = c(50), # Search space for number of trees
  TRAIN_eta = c(0.1),       # Search space for learning ratre
  TRAIN_max_depth = c(4,6,8),    # Search space for tree depth
  TRAIN_window_type =  "FixedWindowCV", #"GrowingWindowCV",
  TRAIN_initial.window = 1e4,    # Window size for training
  TRAIN_horizon = 1e3,           # Future window for testing
  notes="Using AUC as criteria"
)

#-- List of pairs to trade
instruments <- data.table(currs = c("BUY_RES_EURUSD"))

#-- Read the configurations
returns_period = "week" #"month","day" defines the period of aggregating the returns
READ_SELECTED_FEATURES <- F
WRITE_FLAG <- F
SPREAD <- 2 # Spread, make sure there is a file with the specified spread
SL <- config_file$SL[1]
PF <- config_file$PF[1]
test_ratio <- config_file$test_portion[1]
initial.window<-config_file$TRAIN_initial.window[1]
horizon <- config_file$TRAIN_horizon[1]
wind <- config_file$TRAIN_window_type[1]


#------------------------------------------------------------#
############### DEFINE THE FUNCTIONS #########################
#------------------------------------------------------------#

"+" = function(x,y) {
  if(is.character(x) || is.character(y)) {
    return(paste(x , y, sep=""))
  } else {
    .Primitive("+")(x,y)
  }
}

get_sharpe=function(dt_curr,dt_time_lut_prediction_period,PF)
{
  dt_portfolio <-  merge(dt_time_lut_prediction_period,dt_curr,all.x = T,by="index")
  
  #-- Add equity, returns and drawdown
  dt_portfolio[TARGET==1 & decision==1,returns:=PF][TARGET==0 & decision==1,returns:=-1][is.na(returns),returns:=0][,equity:=cumsum(returns)][,drawdown:=cummax(equity)][,drawdown:=(drawdown-equity) ]
  
  mean_returns <- dt_portfolio[,.(mean_returns=sum(returns)),by="ret_per"]
  
  return(list(mean_returns,mean(mean_returns$mean_returns),var(mean_returns$mean_returns),max(dt_portfolio$drawdown)))
  
}

get_mean_returns_and_variance=function(dt_res,dt_time_lut_prediction_period,PF)
{
  
  step <-0.01
  th<-step
  bst_sharpe <- -999
  bst_thr <- 0.01
  
  #--DEBUG
  #th<-0.54
  
  
  
  while(th<0.95)
  {
    dt_curr<-copy(dt_res)
    dt_curr[,decision:=as.numeric(prediction>th)]
    dt_curr <- dt_curr[decision>0.5]
    ret_varg<-get_sharpe(dt_curr,dt_time_lut_prediction_period,PF)
    
    if((ret_varg[[2]]==0 & ret_varg[[3]]==0))
    {
      curr_sharpe<-0
    }else{
      #-- SHARPE RATIO CALCULATION
      #curr_sharpe <- ret_varg[[2]]/sqrt(1+ret_varg[[3]]+ret_varg[[4]])
      #curr_sharpe <- ret_varg[[2]]/(1+sqrt(ret_varg[[3]]))
      #curr_sharpe <- ret_varg[[2]]/(0.01+sqrt(ret_varg[[3]]))
      curr_sharpe <-1-pnorm(0,ret_varg[[2]],sqrt(ret_varg[[3]]))
      
    }
    

    if(curr_sharpe>bst_sharpe)
    {
      bst_sharpe<-curr_sharpe
      bst_thr <- th
      bst_mean_ret <- ret_varg[[2]]
      #bst_var_ret <- sqrt(1+ret_varg[[3]]+ret_varg[[4]])
      #bst_var_ret <- (1+sqrt(ret_varg[[3]]))
      #bst_var_ret <- (0.01+sqrt(ret_varg[[3]]))
      bst_var_ret <- (sqrt(ret_varg[[3]]))
      
    }
    
    th<-th+step
  }
  
  
  return(list(bst_mean_ret,bst_var_ret,bst_thr))
  
}



train_and_predict = function(dt,nrounds,eta,max_depth,initial.window,horizon,target_name="TARGET",index_name="index")
{

  #-- Get feature columns and target columns
  feat_cols <-setdiff(names(dt),target_name)
  target_col <-target_name
  #-- CHeck if index column is there
  index_col_available <- index_name %in% names(dt)
  
  #-- Exclude index from the feature columns
  if(index_col_available)
  {
    feat_cols <- setdiff(feat_cols,index_name)
  }
  
  
  #-- Initialize the resultant table
  dt_res <- data.table(prediction=numeric(0),index=numeric(0),TARGET=numeric(0))
  
  
  i<-1+initial.window
  while(i< (nrow(dt)-horizon-1) )
  {
    
    #-- subset train and prediction and index
    dt_train <- copy(dt[(i-initial.window):i-1,])  
    dt_predict <- copy(dt[i:(i+horizon-1),] )
    
    if(index_col_available)
    {
      dt_index <- copy(dt_predict[,..index_name])
    }
    
    dt_vars_cols_train <- dt_train[,..feat_cols]
    dt_target_train    <- dt_train[,..target_col]
    xgb <- xgboost(data = as.matrix(dt_vars_cols_train), 
                   label = as.matrix(dt_target_train), 
                   eta = eta,
                   max_depth = max_depth, 
                   nround=nrounds,
                   objective = "binary:logistic",
                   #eval_metric = "map",
                   verbose = F
    )
    #print(xgb.importance(model=xgb,feature_names = feat_cols))
    #-- Predict
    y_pred <- predict(xgb,newdata=as.matrix(dt_predict[,..feat_cols]))
    #-- Include predictions
    dt_index<-cbind(dt_index,data.table(prediction=y_pred))  
    #-- Include the ground truth
    dt_index<-cbind(dt_index,dt_predict[,..target_col])  
    dt_res <- rbind(dt_res,dt_index)  
    rm(dt_index)
    cat("\r",round(100.0*i/(nrow(dt)-horizon-1))+"%")
    i<-i+horizon
  }
  
  cat("\n\n")
  return(dt_res) 
}



"+" = function(x,y) {
  if(is.character(x) || is.character(y)) {
    return(paste(x , y, sep=""))
  } else {
    .Primitive("+")(x,y)
  }
}


#-- Sharpe ratio function
# Only the pred is used
# TODO: Use Sortino ratio instead and modify the way the sharpe_ratio is calculated
sharpe_ratio = function(task, model, pred, feats, extra.args) {
  
  predTable <- as.data.table(pred)
#-- Select only the trades we label as true because they build up the portfolio
  predTable <- predTable[response==T]
  if(nrow(predTable)>5)
  {  
  #-- Get the equity and drawdown
  predTable[,equity:=2*(as.numeric(truth)-1.5)][equity>0,equity:=PF][,equity:=cumsum(equity)][,drawdown:=cummax(equity)][,drawdown:=(drawdown-equity) ]
  #-- Calculate the modified sharpe ratio by including the drawdown
    (predTable[nrow(predTable), equity])/((1+max(predTable$drawdown)))
  }else{
    
    (0)
  }
}

#-- Set the sharpe ratio as a custom function for optimizing the models
sharpe_ratio = makeMeasure(
  id = "sharpe_ratio", name = "sharpe_ratio",
  properties = c("classif", "classif.multi", "req.pred",
                 "req.truth"),
  minimize = FALSE,  fun = sharpe_ratio
)

#-- Get the optimal threshold to maximize the portfolio
getBestThresh <- function(dt)
{
  res_orig <- as.data.table(dt)
  thresh_vec <- seq(0.01,0.99,0.01)
  bst_thresh <-0
  max_sharpe_ratio <- -991
  bst_drawdown <- -9999
  max_avg_ret <- -999
  iters <- max(dt$iter)
  
  for (th in thresh_vec)
  {
    res_sel <- copy(res_orig)
    res_sel[,response:=prob.TRUE>th]
    res_sel <- res_sel[response==T]
    if(nrow(res_sel)>10)
    {  
      
      #-- Compute the sharpe ratio as average ret per tra over the variance
      #-- Net equity
      res_sel[,equity:=2*(as.numeric(truth)-1.5)][equity>0,equity:=PF][,equity:=cumsum(equity)][,drawdown:=cummax(equity)][,drawdown:=(drawdown-equity) ]
      total_ret <-  res_sel[nrow(res_sel), equity]
      std_ret <- sqrt(var(res_sel$equity))
      min_drawdown <- max(res_sel$drawdown)
      sharpe_ratio <- total_ret/((1+min_drawdown)*iters)
      
      if(sharpe_ratio>max_sharpe_ratio)
      {
        max_sharpe_ratio <- sharpe_ratio
        max_avg_ret <- total_ret
        bst_thresh <- th
        bst_drawdown <- min_drawdown
        bst_dt <- res_sel
      }
      
    }
  }
  
  return( list(max(bst_dt$drawdown),max_avg_ret, nrow(bst_dt[equity<0]) ,  max_sharpe_ratio, bst_thresh ,bst_dt))
  
}

#------------------------------------------------------------#
###############  READ THE DATA AND FORMAT THE COLUMNS ########
#------------------------------------------------------------#

#-- Read the main labeled dataset
dt<-fread(paste0(data_intermediate_dir,"ML_SL_",SL,"_PF_",PF,"_SPREAD_",SPREAD,"_ALL.csv"))

#-- Read the required features
basic_cols <-fread(data_intermediate_dir+"basic_features.csv")
basic_cols<-basic_cols[grepl("TMS",feats) | grepl("RSI",feats)]
#-- Select the features and the target column
results_cols <- c(names(dt)[grepl("BUY_RES",names(dt)) | grepl("SELL_RES",names(dt)) | grepl("Time",names(dt)) ] )

all_feats_sel<-c(results_cols,basic_cols$feats)
dt<-dt[,..all_feats_sel]

#-- Read the best parameters
best_parameters <-fread(data_input_dir+"best_parameters.csv")

dt[,index:= seq(1,nrow(dt))]
#-- Get the lookup table for time and index
dt_time_lut <- dt[,.(index,Time)]


#------------------------------------------------------------#
###############  Select the latest rows and features #########
#------------------------------------------------------------#

#-- Exclude the last N hours since they might not be available at the time of training the models
end_index <- nrow(dt)-50
start_index <- end_index-initial.window+1
dt<-dt[start_index:end_index]





logical_feats <- names(dt)[grepl("chaikin",names(dt))]
overfit_feats <- names(dt)[grepl("_Open$",names(dt)) | grepl("_High$",names(dt)) | grepl("_Close$",names(dt)) | grepl("_Low$",names(dt))]
non_feat_cols <- c(logical_feats,overfit_feats,"Time",names(dt)[grepl("buy_profit",names(dt))| grepl("^bs_",names(dt)) | grepl("buy_loss",names(dt)) | grepl("sell_loss",names(dt)) | grepl("sell_profit",names(dt))  ] )

exclude_feats <- names(dt)[grepl("EMA",names(dt)) | grepl("williams",names(dt))]

#-- Extract only the feature variables
feat_cols <- setdiff(names(dt),non_feat_cols)
feat_cols <- setdiff(feat_cols,exclude_feats)

dt_sel <- dt[,..feat_cols]

#------------------------------------------------------------#
##########  loop over the currencies, train and save #########
#------------------------------------------------------------#

feats<-basic_cols$feats


i<-1


while(i<(nrow(best_parameters)+1))
{
  curr<-best_parameters[i,instrument]  

  cat(curr+"\n\n")
  
    mdl_name <- strsplit(curr,"_")[[1]][3]+"_"+strsplit(curr,"_")[[1]][1]
  path <- models_dir+mdl_name+"/"
  dt_curr<-copy(dt_sel)
  dt_curr[,TARGET:=.SD >  0.5,.SDcols = curr]
  dt_curr$TARGET <- as.numeric(dt_curr$TARGET)
  
  #-- Extract relevant columns for this run
  train_cols <- c("TARGET",feats)
  dt_curr<-dt_curr[,..train_cols]
  
  print(names(dt_curr[,..feats]))
  
  #-- Train model
  xgb <- xgboost(data = as.matrix(dt_curr[,..feats]), 
                 label = as.matrix(dt_curr$TARGET), 
                 eta = best_parameters[i,eta],
                 max_depth = best_parameters[i,max_depth], 
                 nround=best_parameters[i,nrounds],
                 objective = "binary:logistic",
  #               eval_metric = "auc",
                 verbose = F
  #               early_stopping_rounds = 3
  )
  
  
  #-- Get feature importance
 feat_imp<- as.data.table(xgb.importance(model=xgb,feature_names = feats))
  
  #-- Save the model
  #xgb.save(model=xgb,path+"mdl_"+mdl_name)
  
  #-- Save the feature importance
  #fwrite(feat_imp,path+"featimp_"+mdl_name+".csv")
  
  
  feat_imp_sel<-feat_imp[,.(Feature)]
  setnames(feat_imp_sel,"Feature",curr)
  
  if(i==1)
  {
    feat_imp_all = feat_imp_sel
    
  }else{
    feat_imp_all<-cbind(feat_imp_all,feat_imp_sel)
    
  }
  
  
  cat("#####################\n\n")
  
  
  i<-i+1
}



# DiagrammeR
# xgb.plot.tree



#------------------------ THE END --------------------------------------------------------#



