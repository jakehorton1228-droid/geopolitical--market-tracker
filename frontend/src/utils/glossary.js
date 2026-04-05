/**
 * Glossary — plain-English definitions for technical terms used across the app.
 *
 * Used by InfoTooltip and PageHelp components so that definitions are
 * consistent everywhere a term appears. Edit definitions here, not inline.
 */

export const GLOSSARY = {
  // Event metrics
  goldstein: {
    term: 'Goldstein Scale',
    short: 'Event tone from -10 (extreme conflict) to +10 (extreme cooperation).',
    long: 'A numeric score assigned to each geopolitical event by GDELT. Ranges from -10 (military conflict, violence) to +10 (peace agreements, diplomatic cooperation). A value near 0 means neutral or routine activity.',
  },
  mentions: {
    term: 'Mentions',
    short: 'How many news articles referenced this event.',
    long: 'The number of distinct news articles that mentioned this event. Higher mentions = broader media coverage = more likely to be market-moving.',
  },
  cameo: {
    term: 'CAMEO Code',
    short: 'Standardized event classification used by GDELT.',
    long: 'The Conflict and Mediation Event Observations (CAMEO) taxonomy classifies events into 20 categories like "public statement", "threaten", "assault", "mass violence", etc. The code is a 2-digit number (01-20).',
  },

  // Statistics
  correlation: {
    term: 'Correlation',
    short: 'How closely two things move together. Ranges from -1 to +1.',
    long: 'A statistical measure of how two variables move together. +1 = perfect positive (they always rise together). 0 = no relationship. -1 = perfect inverse (one rises as the other falls). For event-market data, correlations above 0.2 are notable, above 0.5 are strong.',
  },
  pvalue: {
    term: 'p-value',
    short: 'Probability the result is due to random chance. Lower is better.',
    long: 'The p-value measures statistical significance. A p-value below 0.05 means the pattern is unlikely to be random (less than 5% chance). Below 0.01 is very strong evidence. A high p-value means the relationship could just be coincidence.',
  },
  confidenceInterval: {
    term: 'Confidence Interval',
    short: 'The range where the true value likely falls (95% confident).',
    long: 'Shown as shaded bands around a trend line. The wider the band, the less certain the estimate. Narrow bands mean we have enough data to be confident about the number.',
  },
  zscore: {
    term: 'Z-score',
    short: 'How far a value is from normal, in standard deviations.',
    long: 'A measure of how unusual a value is. Z=0 is average. |Z|>2 means the value is in the top/bottom 2.5% of normal. Used to flag anomalies — unusual market moves relative to their recent history.',
  },
  nObservations: {
    term: 'Sample Size (n)',
    short: 'How many data points the statistic is based on.',
    long: 'The number of observations used to calculate a statistic. More data = more reliable. For correlations, n > 100 is minimally reliable, n > 500 is good.',
  },

  // Market metrics
  logReturn: {
    term: 'Log Return',
    short: 'Daily price change, expressed in log form for better math properties.',
    long: 'The natural log of (today\'s price / yesterday\'s price). Used instead of raw percentage change because log returns are symmetric (a -5% move and +5% move have equal magnitude) and sum cleanly over time.',
  },
  dailyReturn: {
    term: 'Daily Return',
    short: 'Percent change in price from the previous day.',
    long: 'Simple percent change: (today - yesterday) / yesterday × 100. A +2% return means the price went up 2%.',
  },
  volatility: {
    term: 'Volatility',
    short: 'How much prices bounce around. Higher = more risk.',
    long: 'Standard deviation of returns. Low volatility means prices are stable; high volatility means prices are swinging a lot, which usually indicates uncertainty or fear in the market.',
  },
  car: {
    term: 'Cumulative Abnormal Return (CAR)',
    short: 'How much more (or less) a stock moved than expected around an event.',
    long: 'Event study metric. Computed by comparing actual returns during an event window to "normal" returns from a pre-event baseline. A positive CAR means the event pushed the price up more than normal.',
  },

  // ML concepts
  accuracy: {
    term: 'Accuracy',
    short: 'Percent of predictions the model got right.',
    long: 'For a binary UP/DOWN prediction, accuracy = (correct predictions / total predictions) × 100. A random guess is 50%. A model beating 55% on market direction is considered useful.',
  },
  auc: {
    term: 'AUC-ROC',
    short: 'How well a model ranks true positives above false positives. 0.5 = random, 1.0 = perfect.',
    long: 'Area Under the ROC Curve. Measures how well the model separates classes across all thresholds. 0.5 = random guessing, 0.7 = decent, 0.8+ = strong, 1.0 = perfect. A better metric than accuracy when classes are imbalanced.',
  },
  featureImportance: {
    term: 'Feature Importance',
    short: 'Which input variables the model relies on most.',
    long: 'Shows which features (event metrics) the model uses most to make predictions. Higher importance = the feature moves the prediction more. Helps you understand what the model is "paying attention to".',
  },
  logisticRegression: {
    term: 'Logistic Regression',
    short: 'A simple, interpretable ML model that predicts probability (0-100%).',
    long: 'One of the oldest and most interpretable ML algorithms. Outputs a probability between 0 and 1, then converts to a class (e.g., UP/DOWN). Each input feature has a coefficient that tells you exactly how it affects the prediction.',
  },
  crossValidation: {
    term: 'Cross-Validation',
    short: 'Testing a model on data it has never seen to check if it generalizes.',
    long: 'Splits the data into folds, trains on some folds, tests on others, repeats. Prevents overfitting — when a model memorizes training data but fails on new data. 5-fold CV means the data is split into 5 parts and the model is tested 5 times.',
  },

  // Data sources
  gdelt: {
    term: 'GDELT',
    short: 'Global Database of Events, Language, and Tone — worldwide geopolitical event data.',
    long: 'An open-source project that monitors news worldwide in 100+ languages and extracts structured event data (who did what to whom, where, when). Updated daily. Our primary source for geopolitical events.',
  },
  fred: {
    term: 'FRED',
    short: 'Federal Reserve Economic Data — US economic indicators.',
    long: 'St. Louis Fed\'s free database of economic time series. We pull GDP, CPI (inflation), unemployment, Fed funds rate, 10-year Treasury yield, and consumer sentiment. Updated monthly or daily depending on the series.',
  },
  polymarket: {
    term: 'Polymarket',
    short: 'A prediction market where people bet on real-world events. Odds = crowd beliefs.',
    long: 'Users buy and sell shares in event outcomes (e.g., "Will Iran and US sign a deal by June?"). The price of a YES share equals the market\'s implied probability (e.g., $0.35 = 35% chance). It\'s the crowd\'s collective forecast.',
  },
  finbert: {
    term: 'FinBERT',
    short: 'An AI model trained to score financial news sentiment.',
    long: 'A BERT transformer fine-tuned on financial news. For each headline, it outputs a sentiment score from -1 (very negative) to +1 (very positive). Better than generic sentiment tools because it understands finance-specific language (e.g., "rate cut" is positive for stocks).',
  },
  rag: {
    term: 'RAG (Retrieval-Augmented Generation)',
    short: 'AI technique where the model is given real data before writing a response.',
    long: 'Instead of asking a language model to generate text from memory (which can hallucinate), RAG first retrieves relevant documents from a database, then passes them as context. The model\'s output is grounded in actual data. Used here to generate intelligence briefings from real headlines and events.',
  },
  langgraph: {
    term: 'LangGraph',
    short: 'A framework for building multi-step AI agents with shared state.',
    long: 'An orchestration library for LLM-powered apps. Instead of one big prompt, you build a graph of specialized nodes (collect data, analyze, write). Each node does one job and updates shared state. Makes complex AI pipelines debuggable and composable.',
  },
};
