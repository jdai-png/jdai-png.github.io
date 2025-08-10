---
title: Non-Parametric Short-Term Market Price Action Forecasting via Historical Pattern Matching
date: 2025-08-10
categories: [Financial Modeling, Time Series Analysis, Machine Learning]
tags: [Market Prediction, Technical Analysis, Pattern Recognition, Non-Parametric Methods]
abstract: This paper explores a novel, non-parametric methodology for forecasting short-term (15-minute bar) market price action based on the identification and analysis of historical patterns of discretized price changes. The proposed approach involves defining a fixed-length pattern window of recent price movements, searching historical data for exact matches of this pattern, and subsequently calculating the empirical probabilities of the next bar's price action based on the observed outcomes following these historical matches. The technical specification for the implementation of this methodology within a software application, potentially driven by a Large Language Model (LLM), is also presented.
image: /assets/img/markov/markov-2.webp
---

## 1. Introduction

The prediction of financial market prices has been a long-standing area of research and a subject of intense interest for investors and traders. Numerous methodologies, ranging from traditional statistical models to sophisticated machine learning algorithms, have been employed in an attempt to forecast future price movements. This paper introduces a conceptually simple yet potentially insightful non-parametric approach that leverages the repetition of historical price patterns to predict the immediate short-term direction of price change. Specifically, we focus on predicting the price action of the next 15-minute trading bar based on patterns observed in the preceding five 15-minute bars.

## 2. Methodology

The proposed methodology involves the following key steps:

### 2.1. Data Acquisition and Transformation

Historical price data, specifically Open, High, Low, and Close prices for a given financial instrument, are required. The data is segmented into 15-minute intervals. For each 15-minute bar, the percentage change from the open price to the close price is calculated using the formula:

$$\text{Percentage Change}_t = \left( \frac{\text{Close}_t - \text{Open}_t}{\text{Open}_t} \right) \times 100$$

A historical window of a specified duration (e.g., 10 trading days) is utilized to build the database of price change patterns.

### 2.2. Pattern Discretization

To facilitate pattern matching, the continuous percentage changes are discretized into a finite set of categorical labels. Predefined bins or thresholds are used to map each percentage change to a specific category. An illustrative set of discretization bins is provided below:

* "Up Big": Percentage Change > 0.5%
* "Up Small": 0.1% ≤ Percentage Change ≤ 0.5%
* "Flat": -0.1% < Percentage Change < 0.1%
* "Down Small": -0.5% ≤ Percentage Change < -0.1%
* "Down Big": Percentage Change < -0.5%

These thresholds can be adjusted based on the specific characteristics and volatility of the asset being analyzed.

### 2.3. Pattern Definition and Matching

A "pattern window" of a fixed length (e.g., 5 consecutive 15-minute bars) is defined. The discretized price changes within this window constitute a "pattern." The current pattern is formed by the discretized price changes of the most recent five 15-minute bars.

A sliding window approach is employed to search the historical data for all instances where a 5-bar sequence exhibits an exact match to the current discretized pattern.

### 2.4. Probabilistic Forecasting

Once all historical occurrences of the current pattern are identified, the price action of the subsequent 15-minute bar (the 6th bar following the matched pattern) is analyzed. The outcome of this next bar is categorized as "Up" (close > open), "Down" (close < open), or "Flat" (close ≈ open, using the same discretization thresholds as above, specifically the "Flat" bin).

The probability of each outcome (Up, Down, Flat) is then calculated empirically based on the frequency of these outcomes following the historical matches:

$$P(\text{Up}) = \frac{\text{Number of times the next bar went Up}}{\text{Total number of pattern matches}}$$

$$P(\text{Down}) = \frac{\text{Number of times the next bar went Down}}{\text{Total number of pattern matches}}$$

$$P(\text{Flat}) = \frac{\text{Number of times the next bar was Flat}}{\text{Total number of pattern matches}}$$

These probabilities represent the forecast for the price action of the next 15-minute bar.

## 3. Technical Specification for Application Integration

To integrate this methodology into a software application, potentially leveraging the capabilities of a Large Language Model (LLM) for user interaction and orchestration, a well-defined technical specification is necessary. The key components and their interfaces are outlined below:

### 3.1. Input Parameters

The application should allow the user to configure the following parameters:

* `timeframe`: Set to '15m' for this specific implementation.
* `lookback_days`: Integer specifying the number of historical days to consider (e.g., 10).
* `pattern_window`: Integer defining the length of the pattern sequence (e.g., 5).
* `discretization_bins`: A configurable set of thresholds for categorizing percentage price changes.

### 3.2. Output Format

The forecasting module should output a structured data format, such as JSON, containing:

* `current_pattern`: An array of the discretized labels representing the current 5-bar pattern.
* `total_matches`: The count of historical occurrences of the `current_pattern`.
* `forecast_probabilities`: A dictionary or object containing the probabilities for "up_probability," "down_probability," and "flat_probability."

### 3.3. LLM Interaction

The LLM can act as an interface for the user to define the analysis parameters and interpret the results. The LLM would:

1.  Receive user requests for a price action forecast based on historical patterns.
2.  Call the underlying pattern-matching module with the specified parameters.
3.  Parse the JSON output from the module.
4.  Present the forecast probabilities and the matched historical pattern to the user in a natural language format.
5.  Inform the user if no matching patterns were found, in which case a forecast cannot be generated.

## 4. Discussion and Conclusion

The non-parametric pattern-matching approach presented in this paper offers a straightforward method for generating short-term market price action forecasts. By discretizing price changes and identifying historical sequences that mirror recent market behavior, empirical probabilities for the next bar's direction can be derived.

This methodology is inherently data-driven and does not rely on assumptions of market efficiency or specific statistical distributions. The accuracy and reliability of the forecasts are directly dependent on the quality and quantity of historical data, as well as the appropriateness of the chosen discretization bins and pattern window length.

Future research could explore the optimization of these parameters, the integration of volume data into the pattern definition, and the comparison of this non-parametric approach with more complex time series forecasting models. Furthermore, the role of LLMs in facilitating user interaction and interpreting the results of such analyses holds significant potential for enhancing the accessibility and understanding of quantitative trading tools.