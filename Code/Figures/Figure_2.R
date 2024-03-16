####Percentage of time spent in microhabitat across summer and winter:
library(lubridate)
library(tidyverse)
library(dplyr)
library(magrittr)
library(dplyr)
library(ggplot2)
library(scales)

df <- read.csv("categorical_data_for_statistics_1_min_filtered.csv",header=T)
df_percentage = df %>%
  group_by(ID, Season, selected_habitat) %>%
  summarise(count = n()) %>%
  mutate(percentage = count / sum(count) * 100) %>%
  select(-count)

p <- ggplot(df_percentage, aes(x = selected_habitat, y = percentage, colour = selected_habitat)) +
  geom_boxplot(outlier.shape = NA) +
  geom_jitter(aes(colour = selected_habitat), width = 0.2, size = 2, alpha = 0.5) +
  facet_wrap(~ Season) + 
  scale_colour_manual(values = c("Bush" = "chartreuse2", "Open" = "orangered1", "Rock" = "azure4")) +
  labs(x = "Microhabitat", y = "Activity (%)", colour = "Microhabitat") +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(colour = "black", fill=NA, size=1),
    axis.ticks.y = element_line(color = "black")
  )

# save your plot
ggsave("Figure_2.jpg", plot = p, width = 6, height = 4, dpi = 300, units = "in", bg = "white")

