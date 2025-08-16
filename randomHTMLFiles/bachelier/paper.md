An academic paper on the ideas from the provided transcript is below.

Deconstructing the Hype: A Re-evaluation of the Black-Scholes Formula in the History of Option Pricing
Abstract

The Black-Scholes model is often hailed as a revolutionary breakthrough in financial economics, earning its creators the Nobel Memorial Prize in Economic Sciences. This paper challenges the conventional narrative by arguing that the celebrated formula was not a novel mathematical invention. Instead, its primary contribution was the establishment of a robust economic framework—the "certainty argument"—that integrated pre-existing mathematical concepts for option pricing with the prevailing economic theories of the time. Drawing on a historical analysis that begins with the pioneering work of Louis Bachelier, this paper repositions the Black-Scholes model as an evolution rather than a revolution, emphasizing that the mathematical heavy lifting had been done decades prior. We will explore the key modifications to Bachelier's original model, including the shift from a normal to a lognormal distribution and the crucial insight on the mean of the distribution provided by John Maynard Keynes. Ultimately, this paper contends that the true genius of Black-Scholes lies in its economic architecture, not its mathematical formulation.

1. Introduction
In the world of quantitative finance, the Black-Scholes formula is often presented as the cornerstone of modern option pricing theory. The 1997 Nobel Prize awarded to Robert C. Merton and Myron S. Scholes for their work on this topic further solidified its legendary status. However, a closer examination of the history of financial mathematics reveals a more nuanced story. The core ideas underpinning the pricing of options did not spring forth fully formed in the 1970s. In fact, the mathematical machinery for such calculations existed long before.


This paper puts forth the argument that the significance of the Black-Scholes model is widely misinterpreted. The true innovation was not the formula for pricing an option—a variation of which was already in use—but the powerful economic argument that made it compatible with the principles of arbitrage-free pricing. This "certainty argument" provided a theoretical justification for using the risk-free rate as the expected return on a stock, thereby creating a universally applicable pricing model. To understand this, we must first travel back to the turn of the 20th century, to the work of a largely forgotten pioneer: Louis Bachelier.

2. The Precursor: Louis Bachelier's Contributions
Long before the names Black and Scholes became synonymous with option pricing, a French mathematician named Louis Bachelier laid the groundwork. In his 1900 doctoral thesis, "Théorie de la Spéculation," Bachelier, an option trader himself, developed a sophisticated mathematical model for speculative markets. Supervised by the great Henri Poincaré, who unfortunately did not appreciate the paper's significance, Bachelier's work was largely ignored for decades. It was only rediscovered and brought to light by a later generation of academics, including Paul Cootner.

Bachelier's model made a groundbreaking assumption: that stock prices follow a Brownian motion, a concept he developed five years before Albert Einstein's famous paper on the topic. For the purpose of pricing an option, he assumed that the stock price follows a normal distribution. The price of a call option, in his view, was simply the expectation of its payoff at expiration.

A call option's payoff is max(S - K, 0), where S is the stock price at expiration and K is the strike price. Bachelier calculated the option price by integrating this payoff function over a Gaussian (normal) probability distribution of S. While his work did not explicitly address the mean of this distribution in a theoretical way, as a trader, he priced options intuitively, and his method was remarkably similar to what traders used in practice.

3. Evolution of the Model: Key Modifications
Bachelier's model, while brilliant, had its limitations. Two major modifications were necessary to transform it into the robust tool used today.

3.1 From Normal to Lognormal Distribution
The first crucial change was the shift from a normal distribution to a lognormal distribution for stock prices. Bachelier's assumption of normality had a significant flaw: it allowed for the possibility of negative stock prices. A stock trading at $100, under a normal distribution, could theoretically drop to a negative value, which is economically impossible.

The lognormal distribution solves this problem. By assuming that the logarithm of the stock price is normally distributed, the stock price itself is always positive. This aligns with the reality of financial markets, where a stock's value can go to zero but not below it. It's worth noting, however, that for certain financial instruments like interest rates, which move in basis points and are not bound by a zero floor in the same way, Bachelier's original normal distribution model can still be more appropriate.

3.2 The Role of the Mean (Drift) and Keynes's Insight
The second, and perhaps more profound, modification concerned the mean of the distribution, or the expected return of the asset. Decades before Black and Scholes, John Maynard Keynes, in his "A Tract on Monetary Reform," provided the key insight. Keynes argued that in an arbitrage-free market, the expected return of an asset over a given period is not a matter of subjective expectation but is determined by the risk-free interest rate.

He illustrated this with a foreign exchange example. To determine the three-month forward price of GBP/USD, one would not simply guess. Instead, one could borrow GBP at the prevailing UK interest rate, convert it to USD, and invest it at the US interest rate. The difference between these rates dictates the forward exchange rate to prevent a risk-free profit.

This concept of an arbitrage relationship was directly applicable to stocks. The expected future price of a stock is its current price compounded at the risk-free rate, adjusted for any dividends (the "carry"). This simple but powerful idea removed the need to forecast the actual return of a stock, a notoriously difficult task.

4. The Black-Scholes Framework: An Economic Revolution
This brings us to the work of Fischer Black, Myron Scholes, and Robert Merton. They did not "discover" a new pricing formula. The mathematical form they used was essentially a lognormal version of Bachelier's formula, with the mean of the distribution defined by the principles Keynes had articulated.

The Nobel-winning contribution was the creation of a risk-neutral framework. They demonstrated that by constructing a portfolio of the underlying asset and a risk-free asset, one could perfectly replicate the payoff of an option. This replication strategy eliminates all market risk, and therefore, the return on this portfolio must be the risk-free rate. This "certainty argument" provided the rigorous economic justification for using the risk-free rate as the drift in the option pricing model.

It was this economic framework that was revolutionary. It allowed for the consistent and objective pricing of options by eliminating the need for subjective expectations about the future performance of the underlying asset. The formula became a consequence of the economic argument, not the other way around.

5. Conclusion
The story of the Black-Scholes model is a powerful lesson in the history of science and the nature of intellectual progress. While the formula itself is an elegant piece of mathematics, its true value lies in the economic reasoning that underpins it. The groundwork was laid by the brilliant but overlooked Louis Bachelier, and key insights were added by thinkers like John Maynard Keynes.

The triumph of Black and Scholes was in synthesizing these ideas into a coherent and economically sound framework that revolutionized the financial industry. By understanding this history, we gain a deeper appreciation for the fact that even the most celebrated formulas are often built on the shoulders of giants, and that true innovation can lie not just in creating something new, but in understanding how to use what already exists in a new and powerful way. 🧐

6. References
Transcript of a discussion on the Black-Scholes formula.

YouTube Video: A discussion of the Black-Scholes formula