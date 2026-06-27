# ============================================================
# PROJECT: Supply Chain Risk & Performance Analytics
# AUTHOR:  Moukthika Vasa
# DATE:    22 June 2026
# PURPOSE: Analyze global supply chain data to identify
#          delivery risk patterns, profitability drivers,
#          and strategic business insights
# DATASET: DataCoSupplyChain — 180,519 orders (2015-2018)
# TOOLS:   R — ggplot2, dplyr, tidyr, lubridate, scales,
#          treemapify, caret (logistic regression)
# OUTPUT:  5 visualizations + predictive model
# GITHUB:  github.com/moukthika-vasa/product-portfolio
# ============================================================

# ============================================================
# SECTION 1: LOAD REQUIRED LIBRARIES
# These packages provide the tools we need for data 
# manipulation, visualization, and modeling
# ============================================================
# Install all required packages (run this ONCE)
install.packages(c(
  "ggplot2",
  "dplyr",
  "tidyr",
  "lubridate",
  "scales",
  "treemapify",
  "caret"
))
library(ggplot2)    # Professional data visualization
library(dplyr)      # Data manipulation and transformation
library(tidyr)      # Data cleaning and reshaping
library(lubridate)  # Date handling and formatting
library(scales)     # Axis formatting (currency, percentage)
library(treemapify) # Treemap visualization
library(caret)      # Machine learning — logistic regression

# ============================================================
# SECTION 2: LOAD DATASET
# Manually select the DataCo Supply Chain CSV file
# Dataset: 180,519 orders | 53 columns | 2015-2018
# ============================================================
df <- read.csv(file.choose(), stringsAsFactors = FALSE)

# Verify dataset loaded correctly
cat("Dataset Dimensions:", nrow(df), "rows x", ncol(df), "columns\n")
# Display the raw date range from the order date column
cat("Date Range:",
    min(df$order.date..DateOrders., na.rm = TRUE),
    "to",
    max(df$order.date..DateOrders., na.rm = TRUE),
    "\n")

# ============================================================
# SECTION 3: INSPECT DATASET STRUCTURE
# Understand the dataset before cleaning or analysis
# ============================================================

# Display structure of the dataset
str(df)

# Display all column names
colnames(df)

# Summary statistics for all variables
summary(df)

# ============================================================
# SECTION 4: DATA PREPARATION & DATE CONVERSION
# Convert date columns from text into proper date-time format
# ============================================================

# Convert order date to POSIXct format
df$order_date <- mdy_hm(df$order.date..DateOrders.)

# Convert shipping date to POSIXct format
df$shipping_date <- mdy_hm(df$shipping.date..DateOrders.)

# Create month field for trend analysis
df$order_month <- floor_date(df$order_date, unit = "month")

# Verify conversion worked correctly
str(df$order_date)

# Display converted date range
cat(
  "Converted Date Range:",
  format(min(df$order_date, na.rm = TRUE), "%Y-%m-%d"),
  "to",
  format(max(df$order_date, na.rm = TRUE), "%Y-%m-%d"),
  "\n"
)

# ============================================================
# SECTION 5: DATA QUALITY ASSESSMENT
# Check missing values and identify columns that are not useful
# for this analysis
# ============================================================

# Count missing values in each column
missing_values <- colSums(is.na(df))

# Display only columns that have missing values
missing_values[missing_values > 0]

# Check the distribution of the late delivery risk target variable
table(df$Late_delivery_risk)

# Check late delivery risk as percentage
prop.table(table(df$Late_delivery_risk)) * 100

# Check unique values in key categorical columns
unique(df$Market)
unique(df$Shipping.Mode)
unique(df$Customer.Segment)
unique(df$Delivery.Status)

# ============================================================
# SECTION 6: REMOVE UNUSED COLUMNS
# Remove fields that provide no analytical value
# ============================================================

# Remove columns that are empty, mostly missing, masked, or not useful
df <- df %>%
  select(
    -Product.Description,
    -Order.Zipcode,
    -Customer.Email,
    -Customer.Password,
    -Product.Image
  )

# Confirm new dataset size after removing unused columns
cat(
  "Dataset Dimensions After Cleanup:",
  nrow(df),
  "rows x",
  ncol(df),
  "columns\n"
)

# ============================================================
# SECTION 7: DEFINE VISUALIZATION COLOR PALETTE
# Standardized color palette for analytical reporting
# and executive-level data visualization
# ============================================================

# Primary colors
dark_slate   <- "#4F6376"
medium_slate <- "#6F8393"
light_slate  <- "#AEB9C2"

# Accent colors
soft_orange  <- "#F3BE73"
teal_accent  <- "#2CA6A4"

# Background and text colors
panel_gray   <- "#EEF1F4"
light_gray   <- "#D9DEE3"
dark_gray    <- "#2F3337"
white        <- "#FFFFFF"

# ============================================================
# SECTION 8: DEFINE VISUALIZATION THEME
# Standard chart formatting applied across all visualizations
# ============================================================

portfolio_theme <- theme_minimal(base_size = 12) +
  
  theme(
    
    plot.title = element_text(
      size = 16,
      face = "bold",
      color = dark_gray
    ),
    
    plot.subtitle = element_text(
      size = 11,
      color = medium_slate
    ),
    
    axis.title = element_text(
      size = 11,
      face = "bold",
      color = dark_gray
    ),
    
    axis.text = element_text(
      color = dark_gray
    ),
    
    panel.grid.minor = element_blank(),
    
    panel.grid.major = element_line(
      color = light_gray,
      linewidth = 0.3
    ),
    
    plot.background = element_rect(
      fill = white,
      color = NA
    ),
    
    panel.background = element_rect(
      fill = white,
      color = NA
    ),
    
    legend.position = "bottom",
    
    legend.title = element_text(
      face = "bold"
    )
  )

# ============================================================
# SECTION 9: CREATE OUTPUT DIRECTORY
# Create a dedicated location for project visualizations,
# model results, and exported analytical outputs
# ============================================================

output_folder <- "project_outputs"

if (!dir.exists(output_folder)) {
  dir.create(output_folder)
}

cat(
  "Output Directory Created:",
  output_folder,
  "\n"
)

# ============================================================
# SECTION 10: PREPARE LATE DELIVERY RISK DATA
# Aggregate late delivery risk by market and shipping mode
# ============================================================

late_risk_summary <- df %>%
  group_by(Market, Shipping.Mode) %>%
  summarise(
    total_orders = n(),
    late_orders = sum(Late_delivery_risk == 1, na.rm = TRUE),
    late_risk_rate = mean(Late_delivery_risk, na.rm = TRUE),
    .groups = "drop"
  )

# Display summarized risk table
print(late_risk_summary)

# Sort by highest risk combinations
late_risk_summary %>%
  arrange(desc(late_risk_rate))

# ============================================================
# SECTION 11: SET SHIPPING MODE DISPLAY ORDER
# Define business-friendly ordering for shipping modes
# ============================================================

late_risk_summary$Shipping.Mode <- factor(
  late_risk_summary$Shipping.Mode,
  levels = c(
    "Same Day",
    "First Class",
    "Second Class",
    "Standard Class"
  )
)

late_risk_summary$Market <- factor(
  late_risk_summary$Market,
  levels = c(
    "Africa",
    "Europe",
    "LATAM",
    "Pacific Asia",
    "USCA"
  )
)

# ============================================================
# SECTION 12: VISUALIZATION 1 - LATE DELIVERY RISK HEATMAP
# Identify high-risk market and shipping mode combinations
# ============================================================

late_risk_summary <- late_risk_summary %>%
  mutate(
    text_color = ifelse(late_risk_rate >= 0.65, white, dark_gray)
  )

late_risk_heatmap <- ggplot(
  data = late_risk_summary,
  mapping = aes(
    x = Shipping.Mode,
    y = Market,
    fill = late_risk_rate
  )
) +
  geom_tile(
    color = white,
    linewidth = 1.1
  ) +
  geom_text(
    aes(
      label = percent(late_risk_rate, accuracy = 0.1),
      color = text_color
    ),
    size = 4.3,
    fontface = "bold"
  ) +
  scale_color_identity() +
  scale_fill_gradientn(
    colors = c(
      "#F4F6F8",
      "#CCD5DD",
      "#8FA2B2",
      "#4F6376",
      "#263F57"
    ),
    values = rescale(c(0.35, 0.45, 0.65, 0.80, 0.97)),
    labels = percent_format(accuracy = 1),
    name = "Late Delivery\nRisk Rate"
  ) +
  labs(
    title = "Late Delivery Risk by Market and Shipping Mode",
    subtitle = "First Class shows the highest risk across every market.",
    x = "Shipping Mode",
    y = "Market",
    caption = "Source: DataCo Supply Chain Dataset"
  ) +
  portfolio_theme +
  theme(
    panel.grid = element_blank(),
    legend.position = "right",
    plot.title = element_text(size = 15, face = "bold", color = dark_gray),
    plot.subtitle = element_text(size = 10, color = medium_slate),
    axis.title = element_text(size = 10.5, face = "bold", color = dark_gray),
    axis.text = element_text(size = 10, color = dark_gray),
    legend.title = element_text(size = 9, face = "bold", color = dark_gray),
    legend.text = element_text(size = 8.5, color = dark_gray),
    plot.margin = margin(18, 24, 18, 18)
  )

print(late_risk_heatmap)

# Save heatmap as a high-resolution image
ggsave(
  filename = file.path(output_folder, "01_late_delivery_risk_heatmap.png"),
  plot = late_risk_heatmap,
  width = 11,
  height = 6,
  dpi = 300,
  bg = "white"
)

# ============================================================
# BUSINESS INSIGHTS
# ------------------------------------------------------------
# • First Class exhibits the highest late delivery risk across
#   all five global markets.
#
# • Standard Class consistently records the lowest late
#   delivery risk, suggesting greater schedule reliability.
#
# • The consistency of these patterns across markets indicates
#   that shipping mode may have a stronger influence on delivery
#   risk than geographic region.
# ============================================================

# ============================================================
# SECTION 13: PREPARE PROFITABILITY AND DISCOUNT DATA
# Prepare order-level fields for discount and profitability analysis
# ============================================================

discount_profit_data <- df %>%
  select(
    Order.Item.Discount.Rate,
    Order.Item.Profit.Ratio,
    Market,
    Customer.Segment
  ) %>%
  filter(
    !is.na(Order.Item.Discount.Rate),
    !is.na(Order.Item.Profit.Ratio)
  )

# Display summary statistics for discount and profit fields
summary(discount_profit_data$Order.Item.Discount.Rate)
summary(discount_profit_data$Order.Item.Profit.Ratio)

# ============================================================
# ANALYTICAL OBSERVATIONS
# ------------------------------------------------------------
# • The median order discount rate is 10%, indicating that
#   most orders receive relatively modest discounts.
#
# • Discount rates range from 0% to 25%, suggesting a
#   controlled pricing strategy rather than aggressive discounting.
#
# • Profit ratios range from -2.75 to 0.50, indicating that
#   while most orders are profitable, some orders generate
#   substantial financial losses.
#
# • The next visualization will examine whether increasing
#   discount rates are associated with declining profitability.
# ============================================================

# ============================================================
# SECTION 14: ANALYZE DISCOUNT-PROFIT RELATIONSHIP
# Measure the relationship between discount rate and profit ratio
# ============================================================

discount_profit_correlation <- cor(
  discount_profit_data$Order.Item.Discount.Rate,
  discount_profit_data$Order.Item.Profit.Ratio,
  use = "complete.obs"
)

cat(
  "Correlation Between Discount Rate and Profit Ratio:",
  round(discount_profit_correlation, 4),
  "\n"
)

# ============================================================
# ANALYTICAL OBSERVATIONS
# ------------------------------------------------------------
# • The correlation between discount rate and profit ratio is
#   approximately zero (-0.0027), indicating virtually no
#   linear relationship between discounting and profitability.
#
# • Higher discount rates do not consistently result in lower
#   profit ratios, suggesting that additional operational
#   factors influence profitability.
#
# • Product mix, shipping efficiency, and fulfillment costs
#   are likely stronger drivers of profit performance than
#   discount strategy alone.
#
# • The next visualization identifies which product categories
#   contribute the most to overall business profitability.
# ============================================================
# ============================================================
# SECTION 15: VISUALIZATION 2 - PRODUCT CATEGORY PROFITABILITY
# Identify the product categories that generate the highest profit
# ============================================================

category_profit_summary <- df %>%
  group_by(Category.Name) %>%
  summarise(
    total_profit = sum(Order.Profit.Per.Order, na.rm = TRUE),
    total_sales = sum(Sales, na.rm = TRUE),
    total_orders = n(),
    .groups = "drop"
  ) %>%
  arrange(desc(total_profit)) %>%
  slice_head(n = 10)

print(category_profit_summary)

category_profit_plot <- ggplot(
  category_profit_summary,
  aes(
    x = reorder(Category.Name, total_profit),
    y = total_profit
  )
) +
  geom_col(
    fill = dark_slate,
    width = 0.65
  ) +
  coord_flip() +
  scale_y_continuous(
    labels = label_dollar(scale_cut = cut_short_scale()),
    limits = c(0, 900000),
    breaks = seq(0, 900000, by = 150000),
    expand = expansion(mult = c(0, 0))
  ) +
  labs(
    title = "Top 10 Most Profitable Product Categories",
    subtitle = "Total profit generated by the ten highest-performing product categories.",
    x = "Product Category",
    y = "Total Profit",
    caption = "Source: DataCo Supply Chain Dataset"
  ) +
  portfolio_theme +
  theme(
    legend.position = "none",
    panel.grid.minor = element_blank(),
    plot.title = element_text(size = 15, face = "bold", color = dark_gray),
    plot.subtitle = element_text(size = 10, color = medium_slate),
    axis.title = element_text(size = 11, face = "bold", color = dark_gray),
    axis.text = element_text(size = 10, color = dark_gray),
    plot.margin = margin(20, 45, 20, 45)
  )

print(category_profit_plot)

# Save Visualization 2
ggsave(
  filename = file.path(
    output_folder,
    "02_top_product_categories_profit.png"
  ),
  plot = category_profit_plot,
  width = 12,
  height = 6.5,
  dpi = 300,
  bg = "white"
)

# ============================================================
# ANALYTICAL OBSERVATIONS
# ------------------------------------------------------------
# • Fishing is the highest-profit product category, followed
#   by Cleats and Camping & Hiking.
#
# • A relatively small number of product categories generate
#   a significant share of total business profit.
#
# • These high-performing categories should receive priority
#   for inventory planning, supplier management, and
#   promotional investments.
#
# • The next visualization examines how revenue is distributed
#   across global markets and product categories.
# ============================================================

# ============================================================
# SECTION 16: PREPARE REVENUE TREEMAP DATA
# Summarize revenue by market and product category
# ============================================================

revenue_treemap_data <- df %>%
  group_by(Market, Category.Name) %>%
  summarise(
    total_sales = sum(Sales, na.rm = TRUE),
    total_orders = n(),
    .groups = "drop"
  ) %>%
  arrange(desc(total_sales)) %>%
  slice_head(n = 25)

print(revenue_treemap_data)

# ============================================================
# SECTION 17: VISUALIZATION 3 - REVENUE TREEMAP
# Identify revenue concentration across markets and categories
# ============================================================

revenue_treemap_plot <- ggplot(
  revenue_treemap_data,
  aes(
    area = total_sales,
    fill = Market,
    label = paste0(
      Category.Name,
      "\n",
      dollar(total_sales, accuracy = 1)
    )
  )
) +
  geom_treemap(
    color = white,
    linewidth = 1
  ) +
  geom_treemap_text(
    color = white,
    place = "center",
    grow = FALSE,
    reflow = TRUE,
    size = 10,
    fontface = "bold"
  ) +
  scale_fill_manual(
    values = c(
      "Africa" = light_slate,
      "Europe" = dark_slate,
      "LATAM" = medium_slate,
      "Pacific Asia" = teal_accent,
      "USCA" = soft_orange
    )
  ) +
  labs(
    title = "Revenue Concentration by Market and Product Category",
    subtitle = "Top revenue-generating market-category combinations across the global supply chain.",
    fill = "Market",
    caption = "Source: DataCo Supply Chain Dataset"
  ) +
  portfolio_theme +
  theme(
    axis.title = element_blank(),
    axis.text = element_blank(),
    panel.grid = element_blank(),
    legend.position = "right",
    plot.title = element_text(size = 15, face = "bold", color = dark_gray),
    plot.subtitle = element_text(size = 10, color = medium_slate),
    legend.title = element_text(size = 9, face = "bold", color = dark_gray),
    legend.text = element_text(size = 8.5, color = dark_gray),
    plot.margin = margin(20, 35, 20, 20)
  )

print(revenue_treemap_plot)

# Save Visualization 3
ggsave(
  filename = file.path(
    output_folder,
    "03_revenue_market_category_treemap.png"
  ),
  plot = revenue_treemap_plot,
  width = 12,
  height = 6.5,
  dpi = 300,
  bg = "white"
)

# ============================================================
# ANALYTICAL OBSERVATIONS
# ------------------------------------------------------------
# • Revenue is concentrated within a limited number of
#   market-category combinations, particularly Fishing and
#   Cleats across LATAM and Europe.
#
# • LATAM and Europe contribute the largest share of revenue
#   among the five global markets analyzed.
#
# • Revenue concentration highlights opportunities to expand
#   successful product categories into additional markets
#   while reducing dependence on a few key segments.
#
# • The next visualization evaluates delivery performance
#   trends over time.
# ============================================================

# ============================================================
# SECTION 18: PREPARE MONTHLY DELIVERY PERFORMANCE DATA
# Summarize late delivery risk by order month
# ============================================================

monthly_delivery_summary <- df %>%
  group_by(order_month) %>%
  summarise(
    total_orders = n(),
    late_orders = sum(Late_delivery_risk == 1, na.rm = TRUE),
    late_risk_rate = mean(Late_delivery_risk, na.rm = TRUE),
    .groups = "drop"
  )

print(monthly_delivery_summary)

# ============================================================
# SECTION 19: VISUALIZATION 4 - MONTHLY DELIVERY PERFORMANCE
# Track late delivery risk over time
# ============================================================

monthly_delivery_plot <- ggplot(
  monthly_delivery_summary,
  aes(
    x = order_month,
    y = late_risk_rate
  )
) +
  geom_line(
    color = dark_slate,
    linewidth = 1.2
  ) +
  geom_point(
    color = soft_orange,
    size = 2.5
  ) +
  scale_y_continuous(
    labels = percent_format(accuracy = 1),
    limits = c(0.50, 0.60)
  ) +
  scale_x_datetime(
    date_labels = "%b %Y",
    date_breaks = "6 months"
  ) +
  labs(
    title = "Monthly Late Delivery Risk Trend",
    subtitle = "Late delivery risk remains consistently above 50% across the analysis period.",
    x = "Order Month",
    y = "Late Delivery Risk Rate",
    caption = "Source: DataCo Supply Chain Dataset"
  ) +
  portfolio_theme +
  theme(
    legend.position = "none",
    panel.grid.minor = element_blank(),
    plot.title = element_text(size = 15, face = "bold", color = dark_gray),
    plot.subtitle = element_text(size = 10, color = medium_slate),
    axis.title = element_text(size = 11, face = "bold", color = dark_gray),
    axis.text = element_text(size = 10, color = dark_gray),
    axis.text.x = element_text(angle = 35, hjust = 1),
    plot.margin = margin(20, 35, 20, 25)
  )

print(monthly_delivery_plot)

# Save Visualization 4
ggsave(
  filename = file.path(
    output_folder,
    "04_monthly_late_delivery_risk_trend.png"
  ),
  plot = monthly_delivery_plot,
  width = 12,
  height = 6.5,
  dpi = 300,
  bg = "white"
)

# ============================================================
# ANALYTICAL OBSERVATIONS
# ------------------------------------------------------------
# • Monthly late delivery risk remained consistently above
#   50% throughout the analysis period.
#
# • Delivery performance shows only modest month-to-month
#   variation, indicating a persistent operational challenge
#   rather than isolated disruptions.
#
# • Sustained improvements in fulfillment processes and
#   logistics execution may be required to reduce delivery
#   risk over the long term.
#
# • The next analysis compares profitability across customer
#   segments.
# ============================================================

# ============================================================
# SECTION 20: PREPARE CUSTOMER SEGMENT PROFITABILITY DATA
# Calculate average profitability by customer segment
# ============================================================

customer_segment_summary <- df %>%
  group_by(Customer.Segment) %>%
  summarise(
    average_profit_ratio = mean(Order.Item.Profit.Ratio, na.rm = TRUE),
    average_sales = mean(Sales, na.rm = TRUE),
    total_orders = n(),
    .groups = "drop"
  ) %>%
  arrange(desc(average_profit_ratio))

print(customer_segment_summary)

# ============================================================
# ANALYTICAL OBSERVATIONS
# ------------------------------------------------------------
# • Consumer, Corporate, and Home Office customers exhibit
#   nearly identical average profit ratios, approximately 12%.
#
# • Customer segment does not appear to be a significant
#   driver of profitability within this dataset.
#
# • Profit improvement initiatives are therefore likely to
#   achieve greater impact by focusing on product categories,
#   pricing strategy, and supply chain operations.
#
# • The next section develops a logistic regression model to
#   predict late delivery risk using operational variables.
# ============================================================

# ============================================================
# SECTION 21: VISUALIZATION 5 - CUSTOMER SEGMENT PROFITABILITY
# Compare average profitability across customer segments
# ============================================================

customer_segment_plot <- ggplot(
  customer_segment_summary,
  aes(
    x = reorder(Customer.Segment, average_profit_ratio),
    y = average_profit_ratio
  )
) +
  geom_col(
    fill = dark_slate,
    width = 0.6
  ) +
  geom_text(
    aes(label = percent(average_profit_ratio, accuracy = 0.1)),
    vjust = -0.4,
    size = 5,
    fontface = "bold",
    color = dark_gray
  ) +
  scale_y_continuous(
    labels = percent_format(accuracy = 1),
    limits = c(0, 0.14),
    expand = expansion(mult = c(0, 0.02))
  ) +
  labs(
    title = "Average Profitability by Customer Segment",
    subtitle = "Average profit ratios remain consistent across Consumer, Corporate, and Home Office customers.",
    x = "Customer Segment",
    y = "Average Profit Ratio",
    caption = "Source: DataCo Supply Chain Dataset"
  ) +
  portfolio_theme +
  theme(
    legend.position = "none",
    panel.grid.minor = element_blank(),
    plot.title = element_text(size = 15, face = "bold", color = dark_gray),
    plot.subtitle = element_text(size = 10, color = medium_slate),
    axis.title = element_text(size = 11, face = "bold", color = dark_gray),
    axis.text = element_text(size = 11, color = dark_gray)
  )

print(customer_segment_plot)

# Save Visualization 5
ggsave(
  filename = file.path(
    output_folder,
    "05_customer_segment_profitability.png"
  ),
  plot = customer_segment_plot,
  width = 12,
  height = 6.5,
  dpi = 300,
  bg = "white"
)

# ============================================================
# ANALYTICAL OBSERVATIONS
# ------------------------------------------------------------
# • Consumer, Corporate, and Home Office customers exhibit
#   nearly identical average profit ratios, approximately 12%.
#
# • Customer segment does not appear to be a significant
#   driver of profitability within this dataset.
#
# • Profit improvement initiatives are therefore likely to
#   achieve greater impact by focusing on product categories,
#   pricing strategy, and supply chain operations.
#
# • The next section develops a logistic regression model to
#   predict late delivery risk using operational variables.
# ============================================================

# ============================================================
# SECTION 22: PREPARE DATA FOR LOGISTIC REGRESSION
# Create modeling dataset to predict late delivery risk
# ============================================================

model_data <- df %>%
  select(
    Late_delivery_risk,
    Shipping.Mode,
    Market,
    Customer.Segment,
    Category.Name,
    Order.Item.Discount.Rate
  ) %>%
  na.omit()

model_data$Late_delivery_risk <- factor(
  model_data$Late_delivery_risk,
  levels = c(0, 1),
  labels = c("No_Late_Risk", "Late_Risk")
)

# Verify model dataset structure
str(model_data)

# Check target variable distribution
table(model_data$Late_delivery_risk)

# ============================================================
# SECTION 23: BUILD LOGISTIC REGRESSION MODEL
# Predict the probability of late delivery risk
# ============================================================

late_delivery_model <- glm(
  Late_delivery_risk ~
    Shipping.Mode +
    Market +
    Customer.Segment +
    Category.Name +
    Order.Item.Discount.Rate,
  data = model_data,
  family = binomial(link = "logit")
)

# Display model summary
summary(late_delivery_model)

# ============================================================
# ANALYTICAL OBSERVATIONS
# ------------------------------------------------------------
# • Shipping mode is the strongest predictor of late delivery
#   risk, with all shipping methods showing statistically
#   significant effects compared with the reference category.
#
# • Market and customer segment are not statistically
#   significant predictors, indicating that delivery risk is
#   driven more by operational factors than geography or
#   customer type.
#
# • Discount rate is not a significant predictor of late
#   delivery risk, reinforcing earlier findings that pricing
#   strategy has little influence on delivery performance.
#
# • The next step evaluates the predictive performance of the
#   logistic regression model using a confusion matrix and
#   classification accuracy.
# ============================================================

# ============================================================
# SECTION 25: MODEL PREDICTIONS AND CONFUSION MATRIX
# Evaluate logistic regression classification performance
# ============================================================

# Generate predicted probabilities
predicted_probability <- predict(
  late_delivery_model,
  type = "response"
)

# Convert probabilities into class predictions
predicted_class <- ifelse(
  predicted_probability >= 0.50,
  "Late_Risk",
  "No_Late_Risk"
)

predicted_class <- factor(
  predicted_class,
  levels = levels(model_data$Late_delivery_risk)
)

# Confusion Matrix
confusionMatrix(
  predicted_class,
  model_data$Late_delivery_risk
)

# ============================================================
# ANALYTICAL OBSERVATIONS
# ------------------------------------------------------------
# • The logistic regression model achieved an overall
#   classification accuracy of 69.54%, outperforming the
#   baseline prediction rate of 54.83%.
#
# • The model demonstrates strong capability in identifying
#   orders without late delivery risk (Sensitivity = 88.29%),
#   while prediction of late deliveries remains more
#   challenging (Specificity = 54.10%).
#
# • A balanced accuracy of 71.19% indicates reasonable
#   predictive performance for operational decision support.
#
# • The next section identifies which operational variables
#   contribute most to predicting late delivery risk.
# ============================================================

# ============================================================
# SECTION 25: VARIABLE IMPORTANCE
# Rank predictors influencing late delivery risk
# ============================================================

variable_importance <- varImp(
  late_delivery_model,
  scale = FALSE
)

print(variable_importance)

# ============================================================
# SECTION 26: PREPARE VARIABLE IMPORTANCE DATA
# Rank the most influential predictors of late delivery risk
# ============================================================

importance_df <- as.data.frame(variable_importance)

importance_df$Variable <- rownames(importance_df)

colnames(importance_df)[1] <- "Importance"

importance_df <- importance_df %>%
  mutate(
    Variable = gsub("Shipping.Mode", "Shipping Mode: ", Variable),
    Variable = gsub("Category.Name", "Category: ", Variable),
    Variable = gsub("Customer.Segment", "Customer Segment: ", Variable),
    Variable = gsub("Market", "Market: ", Variable),
    Variable = gsub("Order.Item.Discount.Rate", "Discount Rate", Variable)
  ) %>%
  arrange(desc(Importance)) %>%
  slice_head(n = 5)

print(importance_df)

# ============================================================
# SECTION 27: VISUALIZATION 6 - MODEL VARIABLE IMPORTANCE
# Visualize the strongest predictors of late delivery risk
# ============================================================

importance_plot <- ggplot(
  importance_df,
  aes(
    x = reorder(Variable, Importance),
    y = Importance
  )
) +
  geom_col(
    fill = dark_slate,
    width = 0.65
  ) +
  coord_flip() +
  scale_y_continuous(
    expand = expansion(mult = c(0, 0.10))
  ) +
  labs(
    title = "Top Predictors of Late Delivery Risk",
    subtitle = "Shipping mode is the dominant operational driver in the logistic regression model.",
    x = "Predictor",
    y = "Model Importance Score",
    caption = "Source: Logistic Regression Model"
  ) +
  portfolio_theme +
  theme(
    legend.position = "none",
    panel.grid.minor = element_blank(),
    plot.title = element_text(size = 15, face = "bold", color = dark_gray),
    plot.subtitle = element_text(size = 10, color = medium_slate),
    axis.title = element_text(size = 11, face = "bold", color = dark_gray),
    axis.text = element_text(size = 10, color = dark_gray),
    plot.margin = margin(20, 55, 20, 35)
  )

print(importance_plot)

# ============================================================
# SECTION 29: PROJECT CONCLUSIONS
# ============================================================

# ============================================================
# PROJECT SUMMARY
# ------------------------------------------------------------
# • First Class shipping exhibited the highest late delivery
#   risk across all global markets, making shipping mode the
#   strongest operational driver of delivery performance.
#
# • Monthly late delivery risk remained consistently above
#   50%, indicating persistent operational inefficiencies
#   rather than isolated disruptions.
#
# • Discount rate showed virtually no relationship with
#   profitability, suggesting that operational efficiency
#   contributes more to business performance than pricing
#   strategy alone.
#
# • Fishing, Cleats, and Camping & Hiking generated the
#   highest total profit and should be prioritized for
#   inventory planning and capacity allocation.
#
# • Customer profitability remained nearly identical across
#   Consumer, Corporate, and Home Office segments, indicating
#   limited value in segment-specific pricing strategies.
#
# • Logistic regression achieved 69.54% classification
#   accuracy and identified shipping mode as the strongest
#   predictor of late delivery risk.
# ============================================================

# ============================================================
# SECTION 30: BUSINESS RECOMMENDATIONS
# ============================================================

# ============================================================
# STRATEGIC RECOMMENDATIONS
# ------------------------------------------------------------
# • Prioritize operational improvements for First Class
#   fulfillment to reduce delivery risk.
#
# • Allocate inventory and logistics resources toward
#   high-profit product categories.
#
# • Focus improvement initiatives on operational efficiency
#   rather than discount strategies.
#
# • Deploy predictive analytics within order management
#   systems to identify high-risk shipments before dispatch.
# ============================================================