# Tracked Symbols

33 financial instruments across 5 asset classes, selected for their sensitivity to geopolitical events.

## Commodities (8)

| Symbol | Name | Why It's Tracked |
|--------|------|------------------|
| CL=F | Crude Oil (WTI) | Most geopolitically sensitive commodity. Reacts to conflict in Russia, Saudi Arabia, Iran, Venezuela, Nigeria, and Israel. |
| BZ=F | Brent Crude | International oil benchmark. Sensitive to Middle East tensions (Saudi Arabia, Iran). |
| GC=F | Gold | Safe-haven asset. Spikes during military conflict, especially Iran and Israel tensions. |
| SI=F | Silver | Precious metal with industrial demand. Correlated with gold during geopolitical stress. |
| NG=F | Natural Gas | Heavily exposed to Russia-Ukraine dynamics and European energy security. |
| ZW=F | Wheat | Major exports from Ukraine and Russia. Disrupted by Black Sea conflict. |
| ZC=F | Corn | Ukraine is a top exporter. Prices spike during conflict-driven supply disruptions. |
| ZS=F | Soybeans | Sensitive to Brazil, US, and Argentina agricultural/trade policy. |

## Currencies (7)

| Symbol | Name | Why It's Tracked |
|--------|------|------------------|
| EURUSD=X | Euro/USD | Reacts to European geopolitics, particularly Germany, France, Italy, Spain. |
| USDJPY=X | USD/Yen | Safe-haven currency. Strengthens during global risk-off events. |
| GBPUSD=X | British Pound/USD | Sensitive to UK political and economic events. |
| USDCNY=X | USD/Chinese Yuan | Reacts to US-China tensions, trade policy, and Chinese domestic events. |
| USDRUB=X | USD/Russian Ruble | Directly tied to Russia sanctions, military actions, and energy policy. |
| USDINR=X | USD/Indian Rupee | Sensitive to Indian geopolitical events and regional tensions. |
| USDBRL=X | USD/Brazilian Real | Reacts to Brazilian political instability and commodity demand shifts. |

## ETFs (14)

| Symbol | Name | Why It's Tracked |
|--------|------|------------------|
| SPY | S&P 500 | Broad US market benchmark. Baseline for measuring abnormal returns. |
| QQQ | Nasdaq 100 | Tech-heavy index. Exposed to US, Taiwan, and South Korea (semiconductors). |
| EEM | Emerging Markets | Aggregate EM exposure. Sensitive to China, Brazil, India, Korea, Taiwan. |
| VWO | Emerging Markets (Vanguard) | Alternative EM benchmark for cross-validation. |
| EWZ | Brazil | Single-country ETF. Reacts to Brazilian political events and commodity prices. |
| EWJ | Japan | Single-country ETF. Sensitive to Japan monetary policy and regional tensions. |
| FXI | China Large Cap | Direct China exposure. Reacts to US-China relations and domestic policy. |
| EWG | Germany | European industrial economy. Sensitive to EU policy and energy security. |
| EWT | Taiwan | Semiconductor hub. Highly sensitive to China-Taiwan tensions. |
| EWY | South Korea | Tech/semiconductor exposure. Sensitive to North Korea tensions. |
| INDA | India | Single-country ETF. Reacts to India-Pakistan tensions and domestic events. |
| XLE | Energy Sector | US energy companies. Correlated with oil prices and energy policy. |
| XLF | Financial Sector | US banks/financials. Sensitive to monetary policy and sanctions. |
| GDX | Gold Miners | Leveraged play on gold. Amplifies safe-haven moves during geopolitical stress. |

## Volatility (1)

| Symbol | Name | Why It's Tracked |
|--------|------|------------------|
| ^VIX | CBOE Volatility Index | "Fear gauge." Spikes during any major negative geopolitical event globally. |

## Bonds (3)

| Symbol | Name | Why It's Tracked |
|--------|------|------------------|
| TLT | 20+ Year Treasury | Long-duration safe haven. Reacts to central bank policy and flight-to-safety flows. |
| IEF | 7-10 Year Treasury | Intermediate-duration benchmark. Sensitive to monetary policy signals. |
| HYG | High Yield Corporate | Risk appetite proxy. Sells off during geopolitical uncertainty as credit spreads widen. |

## Country-Asset Mappings

The system uses domain-encoded mappings to link geopolitical events in specific countries to the instruments most likely to react:

| Country | Mapped Assets | Rationale |
|---------|---------------|-----------|
| Russia | USDRUB=X, NG=F, CL=F | Major energy exporter, sanctions target |
| China | USDCNY=X, FXI, EEM | Second-largest economy, trade war exposure |
| Saudi Arabia | CL=F, BZ=F | OPEC leader, oil production |
| Iran | CL=F, BZ=F, GC=F | Oil exporter, regional conflict → safe-haven gold |
| Ukraine | ZW=F, ZC=F, NG=F | Grain exporter, natural gas transit |
| Brazil | USDBRL=X, EWZ, ZS=F | Commodity exporter, political volatility |
| Japan | USDJPY=X, EWJ | Safe-haven currency, monetary policy |
| Germany | EURUSD=X, EWG | EU industrial engine, energy dependency |
| Taiwan | EWT, QQQ | Semiconductor production (TSMC) |
| South Korea | EWY, QQQ | Semiconductor/tech exposure |
| India | USDINR=X, INDA | Regional power, border tensions |
| UK | GBPUSD=X | Political/economic events |
| Israel | GC=F, CL=F | Regional conflict → safe havens and oil |
| Venezuela | CL=F | Oil exporter, political instability |
| Nigeria | CL=F | African oil producer |

## Event-Type Mappings

Certain event types have default asset associations regardless of country:

| Event Type | Default Assets | Rationale |
|------------|---------------|-----------|
| Military Action | GC=F, ^VIX, CL=F | Conflict drives gold, volatility, and oil |
| Violent Conflict | GC=F, ^VIX, CL=F | Same risk-off dynamics |
| Central Bank | TLT, IEF | Monetary policy directly impacts bonds |
| Trade Policy | EEM, SPY | Trade disputes affect broad markets and EM |