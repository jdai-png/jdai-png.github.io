---
layout: post
title: "Building Stratagem: My Journey into AI-Powered Financial Tooling"
date: 2025-07-19 17:38:06 -0400
categories: programming ai finance
tags: [nextjs, firebase, genkit, gemini]
---

*![main landing page](https://i.imgur.com/3aYtCbS.png)*

## The "Why": From Frustration to Creation

Every trader knows the dance. You have one screen for your interactive charts, a browser tab for your brokerage account, another for real-time market data, a feed for financial news, and probably a few Discord or X (formerly Twitter) lists for sentiment analysis. It's a chaotic ballet of context-switching, a digital juggling act where the cost of dropping a ball isn't just wasted time—it's a missed opportunity.

For years, this was my reality. As a developer with a passion for the financial markets, I was perpetually frustrated by the fragmented landscape of trading tools. Each platform was a silo, powerful in its own right but stubbornly unaware of the others. None of them did *exactly* what I needed, and the mental overhead of stitching together a coherent market view was a constant drain.

Then came the Cambrian explosion of generative AI. Suddenly, we had models capable of understanding language, summarizing complex documents, and generating structured data from natural prompts. A question began to form in my mind: Why wasn't there a tool that could be a true analytical partner? A platform that didn't just *show* me data but helped me *interpret* it, that could digest a 10-page earnings report into three bullet points, or even suggest new research avenues I hadn't considered.

That question was the seed from which Stratagem grew. It began as a classic "scratch your own itch" project: to build the unified, intelligent trading and analysis tool I wished existed.

## The "What": A Guided Tour of Stratagem

What started as a simple idea has blossomed into a powerful, full-stack application designed to be a trader's co-pilot. Here’s a snapshot of the key features we've brought to life so far.

#### Integrated Trading & Analysis: Your Financial Command Center

At its heart, Stratagem unifies your trading and analysis workflow. We've built a seamless integration with the Alpaca trading API, allowing you to connect your brokerage account and manage your financial world from a single pane of glass. You can monitor your portfolio, track open positions, and dive deep into market data without ever leaving the app.

To give users an analytical edge, we've built a suite of specialized tools, including:

* **Max Pain Calculator:** An options analysis tool that helps visualize the price point at which the maximum number of options holders would lose money, a key level watched by many traders. We calculate this using the formula for total dollar value of outstanding puts and calls: $C(S_T) = \sum_{i} N_{c,i} \cdot \max(0, S_T - K_i)$ and $P(S_T) = \sum_{j} N_{p,j} \cdot \max(0, K_j - S_T)$, where the point of maximum pain minimizes $C(S_T) + P(S_T)$.
* **Dynamic Volatility Screeners:** Go beyond simple price filters to find stocks exhibiting unusual price or volume behavior.
* **Watchlist Performance Trackers:** See at a glance how your curated lists of stocks are performing over various timeframes.

*![Placeholder for Screenshot: The main Stratagem dashboard, showcasing a user's portfolio value, a list of open positions, and an interactive chart of a selected stock.](https://i.imgur.com/759k8Pz.png)*

#### AI as a Co-pilot: The Intelligent Edge

This is where Stratagem truly sets itself apart. We are leveraging Google's Genkit and the powerful Gemini family of models to build a suite of intelligent features that feel like magic.

* **The Article Analyzer:** Found a dense article from a financial journal or a breaking news report? Just paste the URL. Our AI reads the article, provides a concise summary, identifies all mentioned stock tickers, and performs a sentiment analysis (bullish, bearish, or neutral) on the content.


* **The AI-Powered Screener:** Forget fiddling with dozens of dropdowns and sliders. Our screener lets you describe the stocks you're looking for in plain English. Simply type a query like, *"Show me profitable technology stocks under $50 with high relative volume and a recent analyst upgrade,"* and the AI translates your request into a precise database query, returning a list of matching stocks.

*![Placeholder for Screenshot: The AI Screener interface. The user has typed a natural language query into a search box. Below, a table of stocks is displayed, with columns for price, volume, P/E ratio, etc., that match the query.](https://i.imgur.com/r2kzlN8.png)*

* **The "Willie" Page & Custom UX:** To push the boundaries of AI integration, we created the "Willie Page"—a fun, creative outlet using Replicate's AI models for image generation based on financial concepts. Beyond that, the entire user experience is customizable. Our unique AI-powered theme generator lets you describe a mood or a concept (e.g., "a 1980s retro wave trading desk") and generates a custom color palette and background image for your dashboard.

*![Placeholder for Screenshot: The AI theme generator in action. A user has typed "deep ocean bioluminescence" into a prompt box, and the app's UI has transformed into a theme with deep blues, glowing greens, and a subtle background image of jellyfish.](https://i.imgur.com/YJD5FUG.png)*

## The "How": Forged in Code, Tempered by Challenges

Building Stratagem has been a thrilling and humbling journey. Every feature introduced its own unique set of problems to solve and lessons to learn.

#### The Pain Points (And the Scars to Prove It)

1.  **The Great API Juggling Act:** Integrating multiple third-party APIs is less like connecting Lego bricks and more like conducting a chaotic orchestra. We wrangled Alpaca for market data, Firebase for authentication and database, Google AI for generative text, and Replicate for images. Each came with its own authentication scheme (OAuth, API keys, service accounts), unique data formats, and strict rate limits. The solution was to build a robust, abstracted service layer that acts as a universal translator, but getting there was a major, time-consuming undertaking.

2.  **The WebSocket Abyss:** Real-time data is the lifeblood of any trading application, but it's notoriously difficult to implement correctly. Our live trade updates and news feeds are powered by WebSockets. We battled silent connection drops, complex authentication handshakes over the persistent connection, and the subtle performance killers that can cripple a UI. Mastering React's `useEffect` hook for managing connection lifecycles, with proper cleanup functions and dependency arrays, was a hard-won victory.

3.  **Prompt Engineering is a Real Profession:** Getting a Large Language Model to consistently return reliable, structured JSON is far more art than science. Early on, our AI would hallucinate fields or break the JSON format with conversational fluff. It took dozens, if not hundreds, of iterations to craft prompts that were resilient. The breakthrough came from combining meticulously detailed instructions, few-shot examples (showing the AI exactly what we wanted), and defining a rigid output schema with Zod. This allowed us to validate the AI's output on the server before ever sending malformed data to the client.

*![Placeholder for a code block or screenshot: A snippet of a complex Genkit prompt, showing the system instructions, a few-shot example, and the Zod schema definition for the expected JSON output.](https://i.imgur.com/qBAtcjq.png)*

#### The Gains (The "Aha!" Moments)

1.  **The Sheer Power of a Modern Stack:** Using Next.js with the App Router, Server Components, and especially Server Actions has been a revelation. The ability to co-locate data-fetching logic with the components that use it, and to call server-side functions directly from the client without manually writing API endpoints, streamlined development immensely. The developer experience was simply phenomenal.

2.  **Component-Driven UI is King:** We built our entire front end with ShadCN and Tailwind CSS. This combination enabled us to move at lightning speed. Instead of wrestling with CSS specificity or writing boilerplate, we were composing beautiful, accessible, and consistent components. The result is a professional-grade interface built in a fraction of the time it would have taken traditionally.

3.  **AI as a True Force Multiplier:** The single greatest "gain" was the moment the core AI features started to click. Seeing the Article Analyzer correctly summarize a jargon-filled SEC filing in seconds, or watching the screener instantly translate a complex thought into a filtered list of stocks—that was when the vision felt truly tangible. It confirmed the core hypothesis: AI, when thoughtfully applied, can be an incredibly powerful tool for augmenting and accelerating a human's workflow.

## What's Next on the Horizon for Stratagem?

The foundation is strong, but our roadmap is packed. We're just getting started. In the near future, we plan to introduce:

* **More Advanced Charting & Analysis:** We're working on implementing tools for advanced options analysis, such as Gamma Exposure (GEX) charts, and building a system for backtesting user-defined trading strategies.
* **Collaborative Features:** Imagine shared watchlists where you and your group can add notes, or the ability to instantly share an AI-generated analysis of a stock with a single click.
* **Proactive AI Agents:** The ultimate goal is to evolve our AI from a reactive co-pilot into a proactive research assistant. An agent that can monitor your watchlist and the news, alerting you to significant events, summarizing overnight market movements in Asia, or even suggesting new opportunities based on your stated investment thesis—all before you've had your morning coffee.

Building Stratagem has been an incredible adventure into the future of software development and finance. It’s a testament to how modern tools can empower a small team—or even a single developer—to build something powerful and solve a real-world problem.

Thanks for following the journey. Stay tuned for what comes next.