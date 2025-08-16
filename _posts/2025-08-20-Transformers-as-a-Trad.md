---
layout: post
title: "On the Application of Transformer Models for Financial Time-Series Forecasting: A Case Study on EUR/USD"
author: js.ai
date: 2025-08-10 16:02:24 -0400
description: "An academic paper investigating the efficacy of the Transformer architecture for forecasting EUR/USD prices. We analyze its performance against a naive baseline, highlighting the challenges posed by market efficiency."
categories: [AI, Machine Learning, Quantitative Finance, Transformers]
tags: [artificial intelligence, machine learning, deep learning, transformers, financial markets, forex, PyTorch, python, trading, data science, time series, EMH]
---

## **On the Application of Transformer Models for Financial Time-Series Forecasting: A Case Study on EUR/USD**

### **Abstract**

This paper investigates the efficacy of the Transformer architecture, a neural network model renowned for its success in natural language processing (NLP), for the task of financial time-series forecasting. We apply a custom-built Transformer model to predict the closing price of the Euro/US Dollar (EUR/USD) currency pair using historical price data and derived technical indicators. The methodology encompasses data preprocessing, feature engineering, model implementation using PyTorch, and a rigorous evaluation framework. Performance is measured against a naive forecast baseline using standard error metrics. Our results indicate that while the Transformer model achieves low error values, its performance is only marginally better than the baseline, primarily learning to replicate the previous day's price. This behavior highlights the significant challenge posed by the high signal-to-noise ratio and the quasi-random walk nature of financial markets, as suggested by the Efficient Market Hypothesis. The study serves as a foundational analysis, demonstrating the implementation of Transformers in this domain and critically evaluating a common pitfall, thereby providing a benchmark and a clear direction for future research in more advanced feature engineering and model architectures.

---

### **1. Introduction**

The forecasting of financial market movements is a notoriously complex problem that has attracted substantial academic and commercial interest. Traditional econometric models, such as ARIMA and GARCH, have long been employed but often struggle to capture the non-linear dynamics inherent in financial data. With the advent of machine learning, models like Support Vector Machines (SVMs) and, more notably, Recurrent Neural Networks (RNNs) and their variant, Long Short-Term Memory (LSTM) networks, have shown promise by modeling temporal dependencies.

The introduction of the Transformer architecture by Vaswani et al. (2017) revolutionized the field of NLP. Its core innovation, the **self-attention mechanism**, allows the model to weigh the importance of different elements in a sequence, irrespective of their distance from each other. This capability to capture long-range dependencies without suffering from the vanishing gradient problems of RNNs makes it a compelling candidate for time-series analysis.

This paper poses the following research question: Can a standard Transformer model, when applied to historical price and indicator data, generate meaningful predictions of future currency prices, or does it succumb to the inherent randomness of the market? We aim to provide a transparent and reproducible implementation of a Transformer model for forecasting the EUR/USD exchange rate. The contribution of this work is twofold: first, we provide a detailed methodological blueprint for applying Transformers to financial data; second, we critically analyze the model's performance against a naive baseline, highlighting the practical challenge of distinguishing a true predictive signal from a simple persistence forecast.

---

### **2. Literature Review**

The application of neural networks to financial forecasting is well-established. LSTMs, in particular, became a standard for sequence-based financial tasks due to their ability to retain information over long periods (Fischer & Krauss, 2018). However, their sequential nature limits parallelization and can still be challenged by very long-term dependencies.

The Transformer architecture (Vaswani et al., 2017) overcomes these limitations. Its parallel processing capability and the self-attention mechanism provide a more powerful tool for sequence modeling. The attention mechanism's ability to create a context-rich representation of each data point by "attending" to all other points in the sequence is theoretically ideal for financial markets, where past events, even distant ones, can influence current prices. Recent research has begun to adapt Transformers for general time-series forecasting, with models like the Informer (Zhou et al., 2021) being specifically designed to handle long sequences efficiently. This paper builds upon these concepts, applying a foundational Transformer architecture to scrutinize its out-of-the-box performance on noisy financial data.

---

### **3. Methodology**

Our methodology is divided into three primary stages: data preparation, model architecture definition, and the training and evaluation protocol.

#### **3.1. Data Acquisition and Preprocessing**

* **Data Source**: We use historical daily candlestick data for the EUR/USD currency pair from 2020 to 2023. Each data point contains open, high, low, and close (OHLC) prices.
* **Feature Engineering**: To enrich the input data, we compute several standard technical indicators:
    * **20-period Moving Average (MA20)**: The simple average of the last 20 closing prices.
    * **Bollinger Bands (BB)**: Bands set at two standard deviations above and below the MA20.
    * **Relative Strength Index (RSI)**: A momentum oscillator calculated over 14 periods. The formula is $RSI = 100 - \frac{100}{1 + RS}$, where $RS$ is the average gain over the average loss.
    * **MA20 Slope**: The first derivative of the MA20, approximating its trend.
* **Feature Scaling**: All features (OHLC, RSI, BBs, MA20, MA20 Slope) are normalized to a range of $[0, 1]$ using Min-Max Scaling to ensure stable training. The formula is:
    $$X_{scaled} = \frac{X - X_{min}}{X_{max} - X_{min}}$$
* **Data Splitting**: The dataset is split sequentially into training (80%), validation (10%), and testing (10%) sets. This non-random, chronological split is critical to prevent **look-ahead bias** and simulate a real-world trading scenario where the model is tested on entirely unseen future data. We use an input sequence length of 30 days to predict the next single day's closing price.

#### **3.2. Model Architecture: Time-Series Transformer**

The `TimeSeriesTransformer` is constructed using PyTorch. Its key components are:

* **Positional Encoding**: Since the self-attention mechanism is permutation-invariant, we must inject information about the relative or absolute position of each data point. We use sinusoidal positional encodings, as proposed by Vaswani et al. (2017):
    $$PE_{(pos, 2i)} = \sin(\frac{pos}{10000^{2i/d_{model}}})$$
    $$PE_{(pos, 2i+1)} = \cos(\frac{pos}{10000^{2i/d_{model}}})$$
    where $pos$ is the position in the sequence, $i$ is the dimension, and $d_{model}$ is the embedding dimension.
* **Multi-Head Self-Attention**: This is the core of the Transformer. Instead of a single attention function, the model employs multiple "heads" that learn different aspects of the data in parallel. The attention score is calculated using the **Scaled Dot-Product Attention**:
    $$Attention(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
    where $Q$ (Query), $K$ (Key), and $V$ (Value) are linear projections of the input, and $d_k$ is the dimension of the key vectors. 
* **Feed-Forward Network**: Each encoder layer contains a position-wise feed-forward network applied independently to each position.
* **Hyperparameters**: The model configuration is detailed below.

| Hyperparameter | Value | Rationale |
| :--- | :--- | :--- |
| Feature Size | 9 | OHLC (4), Indicators (5) |
| Number of Layers | 2 | A balance between complexity and risk of overfitting |
| $d_{model}$ | 64 | Dimensionality of the model's internal representations |
| Number of Heads | 8 | Allows parallel attention to different feature subspaces |
| Feedforward Dim | 256 | Dimension of the hidden layer in the FFN |
| Dropout | 0.1 | Regularization to prevent overfitting |

#### **3.3. Training and Evaluation Protocol**

* **Training**: The model is trained for 20 epochs using the **Adam optimizer** with a learning rate of $10^{-3}$. The objective is to minimize the **Mean Squared Error (MSE)** loss function between the predicted and actual closing prices.
* **Baseline Model**: To properly contextualize the Transformer's performance, we compare it against a **naive forecast** (also known as a persistence model). This baseline predicts that the closing price at time $t+1$ will be the same as the closing price at time $t$.
    $$\hat{y}_{t+1} = y_t$$
* **Evaluation Metrics**: Performance on the test set is quantified using:
    * **Mean Squared Error (MSE)**: $MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$
    * **Root Mean Squared Error (RMSE)**: $RMSE = \sqrt{MSE}$
    * **Mean Absolute Error (MAE)**: $MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$

---

### **4. Results**

The trained Transformer model was evaluated on the unseen test set. The quantitative results, compared against the naive forecast baseline, are presented in Table 1. The qualitative performance is shown in Figure 1, which plots the model's predictions against the actual prices.

**Table 1**: Performance Metrics on the Test Set

| Model | MSE | RMSE | MAE |
| :--- | :--- | :--- | :--- |
| Naive Forecast | 0.000185 | 0.01360 | 0.00951 |
| Transformer | **0.000179** | **0.01338** | **0.00932** |

Visually, the predictions in Figure 1 appear to track the actual price movements very closely.

**Figure 1**: A comparison of the Transformer's predicted closing prices (orange) against the real closing prices (blue) on the test dataset.

However, the quantitative results in Table 1 reveal a crucial insight. While the Transformer model achieves slightly lower error metrics across the board, its performance is only marginally better than the naive baseline. The small improvement suggests that the dominant strategy learned by the model is to predict a value very close to the last known input value, which is precisely the logic of the naive forecast.

---

### **5. Discussion**

The results present a classic challenge in quantitative finance. The model appears successful at first glance (Figure 1), but a rigorous comparison to a simple baseline reveals that it offers little true predictive power. This phenomenon can be interpreted through the lens of the **Efficient Market Hypothesis (EMH)**, which posits that asset prices fully reflect all available information. In its weak form, EMH suggests that future price movements cannot be predicted based on past prices, which would follow a "random walk." Our model, by learning to replicate the last price, has effectively learned the primary characteristic of a random walk process.

The model's powerful self-attention mechanism, instead of uncovering a complex, hidden signal, has simply identified the strongest statistical relationship in the data: price persistence from one day to the next. The high signal-to-noise ratio in financial markets means that this persistence is often the most reliable pattern a model can learn without overfitting to noise.

This study's primary limitation is the constrained feature set. Relying solely on price-derived indicators may not provide sufficient context to escape the random walk trap. Future research should focus on three key areas:

1.  **Enriched Feature Sets**: Incorporating alternative data sources such as trading volume, order book data, macroeconomic news (e.g., interest rate announcements), and sentiment analysis from news or social media could provide the novel information required for genuine prediction.
2.  **Advanced Architectures**: Exploring architectures specifically designed for long-sequence time-series forecasting, such as the Informer or Autoformer, may yield better results by more efficiently processing longer historical contexts.
3.  **Sophisticated Evaluation**: Moving beyond simple price prediction to forecast volatility, directional movement (up/down), or risk metrics could be more viable and valuable applications for such models.

---

### **6. Conclusion**

This paper successfully demonstrated the implementation of a Transformer model for forecasting the EUR/USD exchange rate. We provided a comprehensive methodology covering data preparation, model architecture, and a robust evaluation framework. The central finding is that a standard Transformer, while technically functional, struggles to significantly outperform a naive forecast baseline. It predominantly learns to mirror the previous day's price, a behavior consistent with the properties of efficient markets. This result serves not as a failure of the Transformer architecture itself, but as a critical baseline and cautionary tale for practitioners. It underscores that architectural sophistication alone is insufficient to solve financial forecasting; the key likely lies in the integration of diverse, non-price data and the formulation of more nuanced predictive targets.

---

### **7. References**

* Fischer, T., & Krauss, C. (2018). Deep learning with long short-term memory networks for financial market predictions. *European Journal of Operational Research*, *270*(2), 654-669.
* Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems*, *30*.
* Zhou, H., Zhang, S., Peng, J., Zhang, S., Li, J., Xiong, H., & Zhang, W. (2021). Informer: Beyond efficient transformer for long sequence time-series forecasting. *Proceedings of the AAAI Conference on Artificial Intelligence*, *35*(12), 11106-11115.