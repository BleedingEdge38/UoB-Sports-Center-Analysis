# 🏋️ UoB Sports & Fitness Centre — Strategic Pricing & Retention Analytics

![Python](https://img.shields.io/badge/Language-Python-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Techniques](https://img.shields.io/badge/Techniques-Price%20Elasticity%20%7C%20Sentiment%20%7C%20Churn-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)
![Academic](https://img.shields.io/badge/Academic-MSc%20Business%20Analytics%20Capstone%20%7C%20University%20of%20Birmingham-blue)
![Type](https://img.shields.io/badge/Type-Group%20Project%20(Team%20of%205)-lightgrey)

> **Group Project Disclosure:** This was a group capstone project. My individual contributions include: price elasticity modelling (rolling window regression), sentiment-elasticity integration pipeline, churn analysis scripts, and Streamlit decision dashboard development.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Business Problem](#business-problem)
- [Key Results at a Glance](#key-results-at-a-glance)
- [Tech Stack](#️tech-stack)
- [Dataset Overview](#dataset-overview)
- [Methodology](#methodology)
  - [1. Data Preparation](#1-data-preparation)
  - [2. Price Elasticity Modelling](#2-price-elasticity-modelling)
  - [3. Sentiment Analysis](#3-sentiment-analysis)
  - [4. Churn & Retention Analysis](#4-churn--retention-analysis)
  - [5. Sentiment-Elasticity Integration](#5-sentiment-elasticity-integration)
- [Key Findings](#key-findings)
- [Strategic Recommendations](#strategic-recommendations)
- [Decision Dashboard](#decision-dashboard)
- [Limitations & Future Work](#️limitations--future-work)


---

##  Project Overview

This project delivers a **comprehensive pricing strategy and retention analytics framework** for the University of Birmingham Sport & Fitness Centre — a £55 million flagship facility serving students, staff, alumni, and the wider community. The analysis was commissioned as a real-world consultancy engagement, combining econometric demand modelling, NLP-based sentiment analysis, and an interactive pricing dashboard to support evidence-based pricing decisions.

> The centre hosted events at the **2022 Commonwealth Games** and is ranked **6th nationally in BUCS standings**, giving its pricing decisions both financial and reputational significance.

---

##  Business Problem

The centre operates at the intersection of **financial sustainability and member inclusivity**. Three core tensions define the challenge:

1. **Revenue vs. Accessibility** - Can prices rise to meet operational demands without pricing out students and community members?
2. **Segment Heterogeneity** - How differently do Alumni, Students, Staff, and Community members respond to price changes?
3. **Reactive vs. Predictive** - Can usage and sentiment data identify at-risk members *before* they cancel?

---

##  Key Results at a Glance

| Metric | Value |
|--------|-------|
| Total Records Analysed | **400k attendance + 400k financial transactions** |
| Primary Members in Dataset | **193,747** across 7 membership tiers |
| Total Revenue Observed | **£18.4 million** |
| Top 20% customers contributing to revenue (Pareto) | **63.8%** |
| Projected Annual Incremental Revenue (risk-managed) | **£67,000 (+7.6% uplift)** |
| Estimated Controlled Churn Increase | **~2%** |
| Most Price-Sensitive Segment | **Students** (elasticity: -1.8 to -2.3) |
| Most Inelastic Segment | **Staff** (elasticity: -0.3) |
| Alumni Holiday Sensitivity Reduction | **-42%** vs. term-time |
| Churn Early-Warning Signal (pre-cancellation) | **78% of members show usage decline 90 days before cancellation** |
| Primary Cancellation Driver | **Financial constraints (34%)** |
| Positive Sentiment Concentration | **32.4%** of feedback responses |
| Highest Sentiment Segment | **Community Over 65** (mean score: 0.76) |
| Lowest Sentiment Segment | **Staff Non-Members** (mean score: -0.43) |

---

## Tech Stack

| Tool / Library | Purpose |
|----------------|---------|
| **Python** | Core programming language |
| **pandas / NumPy** | Data wrangling and transformation |
| **statsmodels** | Log-log regression, seasonal decomposition, OLS |
| **scikit-learn** | Ridge regression, K-means clustering, StandardScaler |
| **TextBlob** | Sentiment polarity analysis of qualitative feedback |
| **matplotlib / seaborn** | Static visualisations |
| **Plotly** | Interactive charts in the dashboard |
| **Streamlit** | Interactive pricing decision dashboard |
| **pickle** | Data caching between pipeline stages |
| **scipy** | Statistical testing and distribution analysis |

---


##  Dataset Overview

Six datasets were provided by the UoB Sport & Fitness Centre management system, covering **July 2022 – May 2025**:

| Dataset | Records | Content |
|---------|---------|---------|
| `FinancialData25G24.xlsx` | ~150,000 | Membership fees, facility charges, booking revenues |
| `AttData25G24.xlsx` | ~300,000 | Daily access records across 7 entry points (Gym, Pool, 4× Reception Barriers) |
| `BookingsData25G24.xlsx` | — | Squash, fitness classes, alternative & other activity bookings |
| `NPSUpdated25G24.csv` | ~2,500 | Net Promoter Score survey responses |
| `QualitativeFeedback25G24.xlsx` | ~2,500 | Open-text member feedback |
| `CancellationUpdated25G24.xlsx` | ~2,000 | Membership termination records with stated reasons |

**Seven Primary Membership Segments Analysed:**
`Student` · `Staff` · `Alumni` · `Community` · `Associate` · `Community Over 65` · `Staff Non-Member`

---

##  Methodology

### 1. Data Preparation

A multi-stage pipeline (`A_data_preparation.py`) handled:
- **Schema reconciliation** - standardised column naming across 6 disparate datasets
- **Temporal alignment** - all dates converted to consistent datetime objects
- **Member-level aggregation** - attendance consolidated to monthly totals per member
- **Winsorisation** - outlier treatment at 1st/99th percentiles to preserve distribution
- **Feature engineering** - `academic term indicators`, `seasonal dummies`, `rolling averages`, `Online_Preference`, product categories (gym, swim, classes, courts)
- **Missing value handling** - <2% of records; forward-filled demographic attributes

Final output: **850 monthly observations** at membership tier–facility level, suitable for elasticity estimation.

---

### 2. Price Elasticity Modelling

A **log-log regression** specification (`B_dynamic_pricing.py`) was chosen for direct coefficient interpretability as price elasticity:

```
ln(Quantity)ᵢₜ = α + β·ln(Price)ᵢₜ + γ·Xᵢₜ + εᵢₜ
```

Where `β` = price elasticity of demand, `Xᵢₜ` = seasonal controls, academic term dummies, facility characteristics.

- **Ridge Regression** (λ=0.1) applied within each window to address multicollinearity
- **6-month rolling windows** with monthly progression to capture temporal elasticity shifts
- **Seasonal decomposition** via `statsmodels.seasonal_decompose` (additive, weekly periodicity)
- **VIF < 10** threshold enforced for feature selection; **HC3 robust standard errors** used where heteroscedasticity detected

---

### 3. Sentiment Analysis

TextBlob polarity scores (range: -1.0 to +1.0) were computed for all open-text feedback and combined with z-score normalised NPS responses (`F_sentiment_integration.py`) to create a **composite Net Sentiment Index (NSI)**:

| NSI Range | Classification | Share of Responses |
|-----------|---------------|-------------------|
| 0.1 to 1.0 | Positive | 32.4% |
| -0.1 to 0.1 | Neutral | 55.0% |
| -1.0 to -0.1 | Negative | 12.6% |

Word cloud analysis identified **facility quality, staff responsiveness, and equipment availability** as primary negative sentiment drivers — largely operational rather than pricing-related.

---

### 4. Churn & Retention Analysis

`G_churn_analysis.py` analysed 2,031 cancellation records to surface behavioural early-warning signals:

- **78%** of cancelled members exhibit declining facility usage in the **90-day pre-cancellation window**
- Median decline: **3.2 fewer visits** vs. prior 90-day baseline
- Members with **>50% usage decline** account for **67% of associated revenue impact** despite being <25% of the pre-cancellation sample
- **Top cancellation drivers:** Financial constraints (34%), Relocation (22%), Lack of usage (18%), Facility/service concerns (12%)
- **56%** of cancellation drivers are price-actionable; **31%** are service-actionable

---

### 5. Sentiment-Elasticity Integration

`D_sentiment_elasticity.py` combines elasticity estimates with NSI scores to map each segment into a **four-quadrant strategic matrix**:

| Quadrant | Segments | Strategy |
|----------|----------|----------|
| 🟢 High Sentiment + Low Elasticity | Community Over 65, Associates | **Premium Pricing** (8–12% increases) |
| 🟡 Moderate Sentiment + Moderate Elasticity | Alumni, Community | **Value Bundling & Loyalty** |
| 🔵 Low Sentiment + Low Elasticity | Staff | **Service Improvements First** |
| 🔴 Moderate Sentiment + High Elasticity | Students | **Retention-First, Bundle Offers** |

---

##  Key Findings

**Price Elasticity by Segment:**

| Segment | Term-Time Elasticity | Holiday Elasticity | Pricing Implication |
|---------|---------------------|-------------------|---------------------|
| Students | -1.8 to -2.3 | -1.8 | Highly elastic — avoid direct increases |
| Community | -2.1 | -1.4 | Most volatile (range: -0.8 to -3.2) |
| Alumni | -1.2 | **-0.7** | 42% sensitivity drop in holidays  |
| Community Over 65 | -0.5 | -0.4 | Consistently inelastic  |
| Associate | -0.6 | -0.4 | Inelastic — supports premium positioning  |
| Staff | **-0.3** | -0.3 | Most inelastic — but sentiment declining  |

**Revenue Concentration:**
- Top 20% of customers → 63.8% of total revenue (Pareto distribution)
- Revenue peaks in September, aligned with academic calendar
- Average member tenure: **337 days**; Average Revenue Per User: **£1,624**

---

##  Strategic Recommendations

| Segment | Season | Recommended Move | Guardrails |
|---------|--------|-----------------|-----------|
| Alumni | Holidays | **+8–12%** selective increase | Pause if rolling elasticity crosses -1.0 or NSI falls 0.1 |
| Alumni | Term-time | +5–8% in low-risk weeks | Same thresholds; pair with value comms |
| Community Over 65 | All | **+8–12% staged** | Pause if weekly visits ↓5% for 2 consecutive weeks |
| Associate | All | +5–8% | Monitor ARPU vs. visits to validate premium realisation |
| Students | Holidays | **Bundle/targeted -5% discount** | Pause if weekly visits ↓8% or NSI ↓0.1 |
| Students | Term-time | Maintain current pricing | Service/booking fixes first |
| Staff | All | Maintain or +3% micro-move | Pre-conditioned on service improvements |
| Community | Term-time | Maintain + value-adds | Capacity/booking optimisation before price changes |

**Projected Outcome:** £67,000 annual incremental revenue (+7.6% uplift) with controlled ~2% churn increase.

---

##  Decision Dashboard

An interactive **Streamlit dashboard** (`H_dashboard.py`) was built to enable real-time pricing scenario testing without requiring data science expertise:

**Dashboard Features:**
-  **Manual price sliders** per membership tier with instant revenue projection
-  **Revenue Impact Bar Chart** — projected % change per tier
-  **Churn Risk Scatter Plot** — churn risk level vs. price change magnitude
-  **Summary statistics & auto-generated recommendations** per segment
-  **Export to Excel** — full elasticity analysis and churn risk report downloadable on demand
-  **Human-in-the-loop governance** — manual approval required for price changes >5%

**To run the dashboard:**

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the dashboard
streamlit run src/H_dashboard.py
```

##  How to Run the Full Pipeline

 1. Clone the repository
```
git clone https://github.com/BleedingEdge38/uob-sports-centre-pricing-analytics.git
```

 2. Install dependencies
```
pip install -r requirements.txt
```

 3. Place datasets in data/ directory (see data_schema.md)

 4. Run pipeline scripts in order:
```
python src/A_data_preparation.py
python src/B_dynamic_pricing.py
python src/C_behavioral_analysis.py
python src/D_sentiment_elasticity.py
python src/E_revenue_analysis.py
python src/F_sentiment_integration.py
python src/G_churn_analysis.py
```

 5. Launch the dashboard
```
streamlit run src/H_dashboard.py
```
> Note: Update hardcoded file paths to relative paths before running. Raw datasets are not included due to institutional data governance agreements — contact the repository owner for the anonymised schema.

---

## Limitations & Future Work
- Observational design: correlational findings; causal inference not possible without controlled experimentation
- Limited price variation in historical data reduces elasticity precision for certain segments
- Survey response bias may affect sentiment representativeness
- External economic factors (cost-of-living pressures) are partially controlled but not fully isolated

### Future Research Priorities:

- Predictive churn models incorporating external economic indicators (CPI, student loan disbursement cycles)
- Cross-segment substitution effect analysis
- Dynamic peak-period pricing opportunities

Long-term Customer Lifetime Value (CLV) modelling for retention investment prioritisation

Competitive benchmarking across UK higher education fitness facilities


