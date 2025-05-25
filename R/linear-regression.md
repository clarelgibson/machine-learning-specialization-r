# Predict the cost of homes in California
Clare Gibson
2025-05-24

- [Introduction](#introduction)
  - [Packages](#packages)
  - [Data](#data)
- [Exploratory analysis](#exploratory-analysis)
  - [Glimpse](#glimpse)
  - [Head](#head)
  - [Summary](#summary)
- [Process data](#process-data)
  - [Missing data](#missing-data)

# Introduction

This notebook trains a machine learning model that predicts home costs
in California. The overall objective is to predict the value of home
prices using **9 feature variables and 1 target variable**.

## Packages

The code chunk below loads all of the packages required for this
exercise.

``` r
# Load packages
library(here)
library(tidyverse)
library(janitor)
```

## Data

The data for this exercise is contained in a csv file `housing.csv`.

``` r
# Read data into a data frame
housing_df <- read_csv(here("data/src/housing.csv"))
```

# Exploratory analysis

## Glimpse

``` r
# Show column labels and data types
glimpse(housing_df)
```

    Rows: 20,640
    Columns: 10
    $ longitude          <dbl> -122.23, -122.22, -122.24, -122.25, -122.25, -122.2…
    $ latitude           <dbl> 37.88, 37.86, 37.85, 37.85, 37.85, 37.85, 37.84, 37…
    $ housing_median_age <dbl> 41, 21, 52, 52, 52, 52, 52, 52, 42, 52, 52, 52, 52,…
    $ total_rooms        <dbl> 880, 7099, 1467, 1274, 1627, 919, 2535, 3104, 2555,…
    $ total_bedrooms     <dbl> 129, 1106, 190, 235, 280, 213, 489, 687, 665, 707, …
    $ population         <dbl> 322, 2401, 496, 558, 565, 413, 1094, 1157, 1206, 15…
    $ households         <dbl> 126, 1138, 177, 219, 259, 193, 514, 647, 595, 714, …
    $ median_income      <dbl> 8.3252, 8.3014, 7.2574, 5.6431, 3.8462, 4.0368, 3.6…
    $ median_house_value <dbl> 452600, 358500, 352100, 341300, 342200, 269700, 299…
    $ ocean_proximity    <chr> "NEAR BAY", "NEAR BAY", "NEAR BAY", "NEAR BAY", "NE…

## Head

``` r
# Show the first few rows of the df
head(housing_df)
```

    # A tibble: 6 × 10
      longitude latitude housing_median_age total_rooms total_bedrooms population
          <dbl>    <dbl>              <dbl>       <dbl>          <dbl>      <dbl>
    1     -122.     37.9                 41         880            129        322
    2     -122.     37.9                 21        7099           1106       2401
    3     -122.     37.8                 52        1467            190        496
    4     -122.     37.8                 52        1274            235        558
    5     -122.     37.8                 52        1627            280        565
    6     -122.     37.8                 52         919            213        413
    # ℹ 4 more variables: households <dbl>, median_income <dbl>,
    #   median_house_value <dbl>, ocean_proximity <chr>

## Summary

``` r
# Show summary stats
summary(housing_df)
```

       longitude         latitude     housing_median_age  total_rooms   
     Min.   :-124.3   Min.   :32.54   Min.   : 1.00      Min.   :    2  
     1st Qu.:-121.8   1st Qu.:33.93   1st Qu.:18.00      1st Qu.: 1448  
     Median :-118.5   Median :34.26   Median :29.00      Median : 2127  
     Mean   :-119.6   Mean   :35.63   Mean   :28.64      Mean   : 2636  
     3rd Qu.:-118.0   3rd Qu.:37.71   3rd Qu.:37.00      3rd Qu.: 3148  
     Max.   :-114.3   Max.   :41.95   Max.   :52.00      Max.   :39320  
                                                                        
     total_bedrooms     population      households     median_income    
     Min.   :   1.0   Min.   :    3   Min.   :   1.0   Min.   : 0.4999  
     1st Qu.: 296.0   1st Qu.:  787   1st Qu.: 280.0   1st Qu.: 2.5634  
     Median : 435.0   Median : 1166   Median : 409.0   Median : 3.5348  
     Mean   : 537.9   Mean   : 1425   Mean   : 499.5   Mean   : 3.8707  
     3rd Qu.: 647.0   3rd Qu.: 1725   3rd Qu.: 605.0   3rd Qu.: 4.7432  
     Max.   :6445.0   Max.   :35682   Max.   :6082.0   Max.   :15.0001  
     NA's   :207                                                        
     median_house_value ocean_proximity   
     Min.   : 14999     Length:20640      
     1st Qu.:119600     Class :character  
     Median :179700     Mode  :character  
     Mean   :206856                       
     3rd Qu.:264725                       
     Max.   :500001                       
                                          

# Process data

## Missing data

First we take a look at which columns have missing data that could skew
the model.

``` r
# Count NA values in each column
housing_df |> 
  summarise_all(~ sum(is.na(.))) |> 
  pivot_longer(everything())
```

    # A tibble: 10 × 2
       name               value
       <chr>              <int>
     1 longitude              0
     2 latitude               0
     3 housing_median_age     0
     4 total_rooms            0
     5 total_bedrooms       207
     6 population             0
     7 households             0
     8 median_income          0
     9 median_house_value     0
    10 ocean_proximity        0

The only column that contains missing values is \`total_bedrooms\`. It
is not possible for a housing unit to have null bedrooms. We should be
able to impute a value for the observations that are missing by using
the k nearest neighbours approach.
