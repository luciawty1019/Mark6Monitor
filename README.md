# 🎰 Mark Six Lottery Monitor

**Statistical Anomaly Detection and Fairness Analysis in Mark Six Lottery Draws**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)]([https://mark6monitor-luciawty1019.streamlit.app](https://mark6-monitor.streamlit.app/))

## 📋 Overview

This application analyzes **2,533 Mark Six lottery draws from 2008 to 2026** using advanced statistical methods and machine learning to evaluate the fairness of the lottery and detect anomalous patterns.

**Live Demo:** [[https://mark6monitor-luciawty1019.streamlit.app](https://mark6-monitor.streamlit.app/)]

## ✨ Features

### 📊 Dashboard
- View the latest draw results
- Overall statistics (2008-2026)
- Number frequency distribution chart
- Recent draws table with anomaly scores

### 📈 Year-by-Year Analysis
- Draws per year visualization
- Detailed analysis for any selected year
- Yearly number frequency charts

### ⚖️ Fairness Analysis
- **Chi-square Goodness-of-Fit Test** - Tests overall number distribution
- **Monte Carlo Simulation** - 10,000 random draws for confidence intervals
- **Serial Correlation Test** - Checks independence of consecutive draws
- **Number Pair Independence Test** - Tests if number pairs appear randomly
- **Time-Based Cross-Validation** - Validates model stability across time periods
- **Synthetic Data Validation** - Tests false positive rate

### 🔥 Hot & Cold Numbers
- Overall hot/cold numbers (2008-2026)
- Year-by-year hot/cold analysis
- Zero-frequency numbers highlighted

### ⚠️ Anomaly Detection
- Isolation Forest algorithm
- Risk levels (🔴 HIGH, 🟠 MEDIUM, 🟢 LOW)
- Detailed explanations of why draws were flagged
- Anomalies timeline by year

## 📊 Key Findings

| Metric | Value |
|--------|-------|
| Total Draws Analyzed | 2,533 |
| Years Covered | 2008 - 2026 |
| Chi-square P-value | 0.3293 |
| Overall Fairness | ✅ **FAIR** (no global bias detected) |
| Anomalies Found | 253 (9.9% of draws) |
| Most Common Number | #49 (346 times, +11.6%) |
| Least Common Number | #41 (273 times, -11.9%) |

## 🚀 How to Run Locally

### Prerequisites
- Python 3.12 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/luciawty1019/Mark6Monitor.git
cd Mark6Monitor
