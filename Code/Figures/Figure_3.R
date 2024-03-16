library(dplyr)
library(lubridate)

#load the results
library(ncdf4)
df = read.csv("Data/categorical_data_for_statistics_1_min_filtered.csv")
nc <- nc_open("Data/statistical_models_posterior.nc")

# Print information about the file
print(nc)

# Read the bayesian results from the file
betas <- ncvar_get(nc, "posterior/beta")
dim(betas)

vars <- 1:12
names(vars) <- c("intercept", "tground", "summer", "summer_tground", "prev_bush", 
                 "prev_open", "prev_rock", "prev_tground", "prev_bush_summer", 
                 "prev_rock_summer", "prev_open_summer", "prev_summer_tground")

betas_bush = betas[1,,,]
betas_open = betas[2,,,]
betas_rock = betas[3,,,]

get_data_seq = function(season, df, data_column, by=0.2, scaled=F) {
  data = df[df$Season==season, data_column]
  v = seq(from = min(data), to = max(data), by=by)
  if (!scaled){
    return(v)
  } else { #this must be the same as was done in the python code for variable standarizations
    #      browser()
    m = mean(data.frame(df[, data_column])[,1])
    s = 2*sd(data.frame(df[, data_column])[,1])
    return((v - m)/s)
  }
}
tground_bush_winter = get_data_seq("Winter", df, "Temp_Bush") 
tground_open_winter = get_data_seq("Winter", df, "Temp_Open") 
tground_rock_winter = get_data_seq("Winter", df, "Temp_Rock") 
tground_bush_summer = get_data_seq("Summer", df, "Temp_Bush") 
tground_open_summer = get_data_seq("Summer", df, "Temp_Open") 
tground_rock_summer = get_data_seq("Summer", df, "Temp_Rock") 

std_tground_bush_winter = get_data_seq("Winter", df, "Temp_Bush", scaled = T)
std_tground_open_winter = get_data_seq("Winter", df, "Temp_Open", scaled = T)
std_tground_rock_winter = get_data_seq("Winter", df, "Temp_Rock", scaled = T)
std_tground_bush_summer = get_data_seq("Summer", df, "Temp_Bush", scaled = T)
std_tground_open_summer = get_data_seq("Summer", df, "Temp_Open", scaled = T)
std_tground_rock_summer = get_data_seq("Summer", df, "Temp_Rock", scaled = T)

create_predictions = function(beta, factor_indexes=c(), continous_indexes=c(), checked_factor_indexes=c(), checked_continous_indexes=c(), continous_array, nsims=2000){
  l = length(continous_array)
  Predicted.prob = array(0, dim=c(l, nsims))
  Predicted.prob[,] = 0
  # Fill in these vectors: this is clumsy, but it works
  #check significance
  significance = list()
  indexes = c(factor_indexes, continous_indexes)
  for (i in indexes){
    significance[[i]] = 1#!between(0, quantile(beta[i,,], 0.025), quantile(beta[i,,], 0.975))
  }
  # browser()
  for(i in 1:l) {
    for(j in 1:nsims) {
      for (factor in factor_indexes){
        # browser()
        # significance = between(0, quantile(beta[factor,,], 0.025), quantile(beta[factor,,], 0.975))
        Predicted.prob[i,j] = Predicted.prob[i,j] + significance[[factor]]* beta[factor, j, 1]#mean(beta[factor,,])
      }
      for (factor in continous_indexes){
        # significance = between(0, quantile(beta[factor,,], 0.025), quantile(beta[factor,,], 0.975))
        Predicted.prob[i,j] = Predicted.prob[i,j] + significance[[factor]]*continous_array[i]* beta[factor, j, 1]
      }
      # browser()
      for (factor in checked_factor_indexes){
        # browser()
        # significance = between(0, quantile(beta[factor,,], 0.025), quantile(beta[factor,,], 0.975))
        Predicted.prob[i,j] = Predicted.prob[i,j] + beta[factor, j, 1]
      }
      for (factor in checked_continous_indexes){
        #significance = between(0, quantile(beta[factor,,], 0.025), quantile(beta[factor,,], 0.975))
        Predicted.prob[i,j] = Predicted.prob[i,j] + beta[factor, j, 1]*continous_array[i]
      }
    }
    Predicted.prob[i,] = plogis(Predicted.prob[i,]) # Use plogis instead of invlogit
  }
  LCB50 <-apply(Predicted.prob, 1, quantile, prob=0.25)
  UCB50 <-apply(Predicted.prob, 1, quantile, prob=0.75)
  LCB5 <-apply(Predicted.prob, 1, quantile, prob=0.025)
  UCB5 <-apply(Predicted.prob, 1, quantile, prob=0.975)
  average = apply(Predicted.prob, 1, mean)
  return(data.frame(estimate = average, LCB50=LCB50, UCB50=UCB50, LCB5=LCB5, UCB5=UCB5))
}
#winter
result_rock = create_predictions(betas_rock, factor_indexes=c(1), continous_indexes=c(), checked_continous_indexes=c(2), continous_array=std_tground_rock_winter, nsims = 2000)
result_bush = create_predictions(betas_bush, factor_indexes=c(1), continous_indexes=c(), checked_continous_indexes=c(2), continous_array=std_tground_bush_winter, nsims = 2000)
result_open = create_predictions(betas_open, factor_indexes=c(1), continous_indexes=c(), checked_continous_indexes=c(2), continous_array=std_tground_open_winter, nsims = 2000)
#summer
result_rock_summer = create_predictions(betas_rock, factor_indexes=c(1,3), continous_indexes=c(2), checked_continous_indexes=c(4), continous_array=std_tground_rock_summer, nsims = 2000)
result_bush_summer = create_predictions(betas_bush, factor_indexes=c(1,3), continous_indexes=c(2), checked_continous_indexes=c(4), continous_array=std_tground_bush_summer, nsims = 2000)
result_open_summer = create_predictions(betas_open, factor_indexes=c(1,3), continous_indexes=c(2), checked_continous_indexes=c(4), continous_array=std_tground_open_summer, nsims = 2000)

#include prev
### winter
#rock
prev_rock_in_rock = create_predictions(betas_rock, factor_indexes=vars[c("intercept", "prev_rock")], continous_indexes=vars["tground"], checked_continous_indexes=vars["prev_tground"], continous_array=std_tground_rock_winter, nsims = 2000)
prev_open_in_rock = create_predictions(betas_rock, factor_indexes=vars[c("intercept", "prev_open")], continous_indexes=c(), checked_continous_indexes=vars["tground"], continous_array=std_tground_rock_winter, nsims = 2000)
prev_bush_in_rock = create_predictions(betas_rock, factor_indexes=vars[c("intercept", "prev_bush")], continous_indexes=c(), checked_continous_indexes=vars["tground"], continous_array=std_tground_rock_winter, nsims = 2000)
#open
prev_open_in_open = create_predictions(betas_open, factor_indexes=vars[c("intercept", "prev_open")], continous_indexes=vars["tground"], checked_continous_indexes=vars["prev_tground"], continous_array=std_tground_open_winter, nsims = 2000)
prev_rock_in_open = create_predictions(betas_open, factor_indexes=vars[c("intercept", "prev_rock")], continous_indexes=c(), checked_continous_indexes=vars["tground"], continous_array=std_tground_open_winter, nsims = 2000)
prev_bush_in_open = create_predictions(betas_open, factor_indexes=vars[c("intercept", "prev_bush")], continous_indexes=c(), checked_continous_indexes=vars["tground"], continous_array=std_tground_open_winter, nsims = 2000)
#bush
prev_open_in_bush = create_predictions(betas_bush, factor_indexes=vars[c("intercept", "prev_open")], continous_indexes=c(), checked_continous_indexes=vars["tground"], continous_array=std_tground_bush_winter, nsims = 2000)
prev_rock_in_bush = create_predictions(betas_bush, factor_indexes=vars[c("intercept", "prev_rock")], continous_indexes=c(), checked_continous_indexes=vars["tground"], continous_array=std_tground_bush_winter, nsims = 2000)
prev_bush_in_bush = create_predictions(betas_bush, factor_indexes=vars[c("intercept", "prev_bush")], continous_indexes=vars["tground"], checked_continous_indexes=vars["prev_tground"], continous_array=std_tground_bush_winter, nsims = 2000)
##Summer
#rock
prev_rock_in_rock_summer = create_predictions(betas_rock, factor_indexes=vars[c("intercept", "summer", "prev_rock", "prev_rock_summer")], continous_indexes=vars[c("tground", "summer_tground")], checked_continous_indexes=vars["prev_summer_tground"], continous_array=std_tground_rock_summer, nsims = 2000)
prev_open_in_rock_summer = create_predictions(betas_rock, factor_indexes=vars[c("intercept", "summer", "prev_open")], continous_indexes=vars[c("tground", "summer_tground")], checked_factor_indexes =vars["prev_open_summer"], continous_array=std_tground_rock_summer, nsims = 2000)
prev_bush_in_rock_summer = create_predictions(betas_rock, factor_indexes=vars[c("intercept", "summer", "prev_bush")], continous_indexes=vars[c("tground", "summer_tground")], checked_factor_indexes =vars["prev_bush_summer"], continous_array=std_tground_rock_summer, nsims = 2000)
#open
prev_open_in_open_summer = create_predictions(betas_open, factor_indexes=vars[c("intercept", "summer", "prev_open", "prev_open_summer")], continous_indexes=vars[c("tground", "summer_tground")], checked_continous_indexes=vars["prev_summer_tground"], continous_array=std_tground_open_summer, nsims = 2000)
prev_rock_in_open_summer = create_predictions(betas_open, factor_indexes=vars[c("intercept", "summer", "prev_rock")], continous_indexes=vars[c("tground", "summer_tground")], checked_factor_indexes =vars["prev_open_summer"], continous_array=std_tground_open_summer, nsims = 2000)
prev_bush_in_open_summer = create_predictions(betas_open, factor_indexes=vars[c("intercept", "summer", "prev_bush")], continous_indexes=vars[c("tground", "summer_tground")], checked_factor_indexes=vars["prev_bush_summer"], continous_array=std_tground_open_summer, nsims = 2000)
#bush
prev_open_in_bush_summer = create_predictions(betas_bush, factor_indexes=vars[c("intercept", "summer", "prev_open")], continous_indexes=vars[c("tground", "summer_tground")], checked_factor_indexes=vars["prev_open_summer"], continous_array=std_tground_bush_summer, nsims = 2000)
prev_rock_in_bush_summer = create_predictions(betas_bush, factor_indexes=vars[c("intercept", "summer", "prev_rock")], continous_indexes=vars[c("tground", "summer_tground")], checked_factor_indexes=vars["prev_rock_summer"], continous_array=std_tground_bush_summer, nsims = 2000)
prev_bush_in_bush_summer = create_predictions(betas_bush, factor_indexes=vars[c("intercept", "summer", "prev_bush", "prev_bush_summer")], continous_indexes=vars[c("tground", "summer_tground")], checked_continous_indexes=vars["prev_summer_tground"], continous_array=std_tground_bush_summer, nsims = 2000)


#add season and habitat columns
# Add required columns
result_rock$season <- "winter"
result_rock$prev <- "no"
result_bush$season <- "winter"
result_bush$prev <- "no"
result_open$season <- "winter"
result_open$prev <- "no"

result_rock_summer$season <- "summer"
result_rock_summer$prev <- "no"
result_bush_summer$season <- "summer"
result_bush_summer$prev <- "no"
result_open_summer$season <- "summer"
result_open_summer$prev <- "no"

prev_rock_in_rock$season <- "winter"
prev_rock_in_rock$prev <- "rock"
prev_open_in_rock$season <- "winter"
prev_open_in_rock$prev <- "open"
prev_bush_in_rock$season <- "winter"
prev_bush_in_rock$prev <- "bush"

prev_open_in_open$season <- "winter"
prev_open_in_open$prev <- "open"
prev_rock_in_open$season <- "winter"
prev_rock_in_open$prev <- "rock"
prev_bush_in_open$season <- "winter"
prev_bush_in_open$prev <- "bush"

prev_open_in_bush$season <- "winter"
prev_open_in_bush$prev <- "open"
prev_rock_in_bush$season <- "winter"
prev_rock_in_bush$prev <- "rock"
prev_bush_in_bush$season <- "winter"
prev_bush_in_bush$prev <- "bush"

prev_open_in_rock_summer$season <- "summer"
prev_open_in_rock_summer$prev <- "open"
prev_bush_in_rock_summer$season <- "summer"
prev_bush_in_rock_summer$prev <- "bush"
prev_rock_in_rock_summer$season <- "summer"
prev_rock_in_rock_summer$prev <- "rock"

prev_open_in_open_summer$season <- "summer"
prev_open_in_open_summer$prev <- "open"
prev_rock_in_open_summer$season <- "summer"
prev_rock_in_open_summer$prev <- "rock"
prev_bush_in_open_summer$season <- "summer"
prev_bush_in_open_summer$prev <- "bush"

prev_open_in_bush_summer$season <- "summer"
prev_open_in_bush_summer$prev <- "open"
prev_rock_in_bush_summer$season <- "summer"
prev_rock_in_bush_summer$prev <- "rock"
prev_bush_in_bush_summer$season <- "summer"
prev_bush_in_bush_summer$prev <- "bush"

### add habitat
# Winter
result_rock$habitat <- "rock"
result_bush$habitat <- "bush"
result_open$habitat <- "open"

prev_rock_in_rock$habitat <- "rock"
prev_open_in_rock$habitat <- "rock"
prev_bush_in_rock$habitat <- "rock"

prev_open_in_open$habitat <- "open"
prev_rock_in_open$habitat <- "open"
prev_bush_in_open$habitat <- "open"

prev_open_in_bush$habitat <- "bush"
prev_rock_in_bush$habitat <- "bush"
prev_bush_in_bush$habitat <- "bush"

# Summer
result_rock_summer$habitat <- "rock"
result_bush_summer$habitat <- "bush"
result_open_summer$habitat <- "open"

prev_rock_in_rock_summer$habitat <- "rock"
prev_open_in_rock_summer$habitat <- "rock"
prev_bush_in_rock_summer$habitat <- "rock"

prev_open_in_open_summer$habitat <- "open"
prev_rock_in_open_summer$habitat <- "open"
prev_bush_in_open_summer$habitat <- "open"

prev_open_in_bush_summer$habitat <- "bush"
prev_rock_in_bush_summer$habitat <- "bush"
prev_bush_in_bush_summer$habitat <- "bush"

### Add temperature
# Winter
result_rock$temp <- tground_rock_winter
result_bush$temp <- tground_bush_winter
result_open$temp <- tground_open_winter

prev_rock_in_rock$temp <- tground_rock_winter
prev_open_in_rock$temp <- tground_rock_winter
prev_bush_in_rock$temp <- tground_rock_winter

prev_open_in_open$temp <- tground_open_winter
prev_rock_in_open$temp <- tground_open_winter
prev_bush_in_open$temp <- tground_open_winter

prev_open_in_bush$temp <- tground_bush_winter
prev_rock_in_bush$temp <- tground_bush_winter
prev_bush_in_bush$temp <- tground_bush_winter

# Summer
result_rock_summer$temp <- tground_rock_summer
result_bush_summer$temp <- tground_bush_summer
result_open_summer$temp <- tground_open_summer

prev_rock_in_rock_summer$temp <- tground_rock_summer
prev_open_in_rock_summer$temp <- tground_rock_summer
prev_bush_in_rock_summer$temp <- tground_rock_summer

prev_open_in_open_summer$temp <- tground_open_summer
prev_rock_in_open_summer$temp <- tground_open_summer
prev_bush_in_open_summer$temp <- tground_open_summer

prev_open_in_bush_summer$temp <- tground_bush_summer
prev_rock_in_bush_summer$temp <- tground_bush_summer
prev_bush_in_bush_summer$temp <- tground_bush_summer

all_results <- rbind(result_rock, result_bush, result_open, 
                     result_rock_summer, result_bush_summer, result_open_summer,
                     prev_rock_in_rock, prev_open_in_rock, prev_bush_in_rock,
                     prev_open_in_open, prev_rock_in_open, prev_bush_in_open,
                     prev_open_in_bush, prev_rock_in_bush, prev_bush_in_bush,
                     prev_rock_in_rock_summer, prev_open_in_rock_summer, prev_bush_in_rock_summer,
                     prev_open_in_open_summer, prev_rock_in_open_summer, prev_bush_in_open_summer,
                     prev_open_in_bush_summer, prev_rock_in_bush_summer, prev_bush_in_bush_summer)

all_results$prev = as.factor(all_results$prev)
# Calculate the minimum and maximum temperature for each season
min_temp_summer <- min(all_results$temp[all_results$season == "summer"], na.rm = TRUE)
max_temp_summer <- max(all_results$temp[all_results$season == "summer"], na.rm = TRUE)
min_temp_winter <- min(all_results$temp[all_results$season == "winter"], na.rm = TRUE)
max_temp_winter <- max(all_results$temp[all_results$season == "winter"], na.rm = TRUE)

all_results$estimate <- all_results$estimate * 100
all_results$LCB50 <- all_results$LCB50 * 100
all_results$UCB50 <- all_results$UCB50 * 100

# Plot
library(ggplot2)

p = ggplot(all_results, aes(x = temp, y = estimate, color = prev)) +
  geom_line(size=1.15) +
  geom_line(aes(x = temp, y = LCB50), linetype = "11") +
  geom_line(aes(x = temp, y = UCB50), linetype = "11") +
  facet_grid(habitat ~ season, scales = "free_x", labeller = labeller(.cols = tools::toTitleCase, .rows = tools::toTitleCase)) +
  labs(x = "Temperature (°C)", y = "Probability of Mircohabitat Selection (%)" , color = "Previous microhabitat") +
  theme_minimal() +
  theme(panel.grid = element_blank(),
        panel.border = element_rect(colour = "black", fill=NA, size=1),
        axis.ticks = element_line(colour = "black"),
        strip.text.x = element_text(size = 12),  # Increase top label size
        strip.text.y = element_text(size = 12),  # Increase side label size
        legend.title = element_text("Previous microhabitat"), legend.position = "top") +
  scale_color_manual(values = c("bush" = "green", "open" = "orange", "rock" = "grey10", "no" = "blue"),
                     labels = c("bush" = "Bush", "open" = "Open", "rock" = "Rock", "no" = "None"),
                     breaks = c("no", "open", "bush", "rock")) +
  guides(color = guide_legend(nrow = 2, order = 1,))

ggsave("Figure_3.jpg", width = 5, height = 7, dpi = 300, bg = "white")
