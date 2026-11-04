<div align="center">

# Hey, I'm Thinuri 👋

### A little maths. A lot of curiosity. Plenty of data.

🧠 Brains &nbsp; · &nbsp; 🎲 Statistics &nbsp; · &nbsp; 🤖 Machine learning &nbsp; · &nbsp; 🤝 People

![Neuroimaging](https://img.shields.io/badge/Neuroimaging-A78BFA?style=for-the-badge&logoColor=white)
![Statistics](https://img.shields.io/badge/Statistics-38BDF8?style=for-the-badge&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine_Learning-FBBF24?style=for-the-badge&logoColor=black)
![Health Research](https://img.shields.io/badge/Health_Research-F472B6?style=for-the-badge&logoColor=white)

**Thinuri Welithotage · University of Toronto**  
Mathematics and Its Applications · Statistics and Probability

[🧠 Brain lab](#brain-lab) · [🎲 Data lab](#data-lab) · [🤖 Work & AI](#work-lab) · [🛠️ Applied models](#applied-lab) · [🤝 Connection](#connection-lab)

</div>

---

## 🧭 Pick a rabbit hole

My projects explore how brains change, why data goes missing, how work affects people, and what models can tell us about it all.

| If you're curious about… | Start here |
| :--- | :--- |
| 🧠 Brain changes and MRI biomarkers | [Neuroimaging projects](#brain-lab) |
| 🎲 Biased estimates and missing surveys | [Longitudinal data projects](#data-lab) |
| 🤖 AI exposure and workplace stress | [Work and AI projects](#work-lab) |
| 🛠️ Hospitals and insurance pricing | [Applied modelling](#applied-lab) |
| 🤝 Loneliness and everyday interactions | [Social connection projects](#connection-lab) |

> **Explore at your own pace:** open a project's details for methods, dates, and research context. “Explore project” links lead to materials already in this repository.

<a id="brain-lab"></a>

## 🧠 Brains, scans & hidden signals

### 🧠 Mapping how brains change

**Longitudinal MRI-Based Modelling of Neurodegenerative Disease Progression**

Following brain trajectories over time to understand neurodegenerative disease.

<details>
<summary>🔎 Open the research notebook</summary>

**Baycrest Academy for Research · Wang Lab, Rotman Research Institute · May 2026–Present**  
SUDS Scholar (May–August 2026), continuing as a Research Volunteer (September 2026–Present). Co-supervised by Dr. Lei Wang and Dr. Yann Cobigo.

- Developed voxelwise Bayesian mixed-effects models in logit space, mapping fitted trajectories back to grey-matter space to study behavioural variant frontotemporal dementia (bvFTD).
- Generated brain maps showing the rate of tissue loss and how it changes over time.
- Used Linux-based high-performance computing clusters for computationally intensive MRI analyses.
- Developing an interactive FSLeyes plugin to inspect individual grey-matter density, velocity, and acceleration trajectories alongside brain maps.

**Research output:** Generated voxelwise 4D maps of grey-matter density, velocity, and acceleration to explore how the location and rate of tissue change evolve over time.

**Current stage:** Exploratory disease-trajectory modelling. Conversion prediction and quantitative comparison with a linear baseline remain to be validated.

Presented at the SUDS Research Showcase, University of Toronto, in August 2026.

</details>

### 🧠 Finding clues to Alzheimer's

**Alzheimer's Biomarker Detection Using MRI**

Turning MRI scans into radiomic features and testing machine learning models.

**My contribution:** Implemented the analysis code, contributed to the paper write-up, and delivered the three-minute talk and poster presentation.

<details>
<summary>🔎 Open the research notebook</summary>

**STEM Fellowship · National Inter-University Health Data & AI Program · May–August 2025**

- Built machine learning models using scikit-learn and PyTorch to investigate early-stage Alzheimer's biomarkers from MRI data.
- Extracted radiomic features from OASIS-2 structural and longitudinal MRI data using PyRadiomics.
- Processed neuroimaging data using NiBabel, Nilearn, and SimpleITK.
- Evaluated model performance using cross-validation and feature selection.

Presented through the 2025 STEM Fellowship program and at the University of Toronto's Arts and Science Union Undergraduate Research Conference in March 2026.

[Explore project](./Alzheimer_Biomarkers/) · [Watch presentation](https://www.youtube.com/watch?v=W4WdSREN0iU) · [Conference paper](https://doi.org/10.48448/pby5-a658)

</details>

---

<a id="data-lab"></a>

## 🎲 When data gets complicated

### 🎲 When follow-up isn't random

**Bias in Linear Mixed Models with Outcome-Dependent Follow-up**

Testing what happens to statistical inference when earlier outcomes influence who gets observed next.

**My contribution:** Contributed to writing the manuscript.

<details>
<summary>🔎 Open the research notebook</summary>

**University of Toronto · January–April 2026**  
Supervised by Dr. Omidali (Omid) Aghababaei Jazi.

- Developed a simulation framework in which observation times depend on prior outcomes.
- Modelled longitudinal data using linear mixed-effects models with random intercepts and slopes.
- Evaluated estimation bias, mean squared error, and confidence interval coverage when standard ignorability assumptions are violated.

</details>

### 🎲 Will tomorrow's survey get answered?

**Simulation and Machine Learning for Longitudinal Workplace Survey Data**

1,000 simulated workers. 90 days. A prediction problem about missing responses.

**My contribution:** Implemented the full project and am currently preparing the paper write-up.

<details>
<summary>🔎 Open the research notebook</summary>

**University of Toronto · ReSTORE Lab · February 2026–Present**

- Developed a reproducible synthetic dataset representing workplace conditions, stress-related measures, and survey participation across 1,000 simulated workers over 90 days.
- Evaluated GRU sequence models alongside logistic regression and personal response-history baselines to predict next-day survey nonresponse.
- Assessed performance using worker-separated evaluation, additional split and training runs, calibration diagnostics, and checks for information leakage.

</details>

---

<a id="work-lab"></a>

## 🤖 Work, people & AI

### 🤖 What makes a job vulnerable to AI?

**Occupational Vulnerability to AI-Driven Change**

664 occupations × 128 skill features, with psychosocial protection in the picture.

**My contribution:** Implemented the full analysis and wrote the paper.

<details>
<summary>🔎 Open the research notebook</summary>

**University of Toronto · ReSTORE Lab · September 2025–February 2026**

Examined how skill composition, task structure, and psychosocial buffers shape occupational vulnerability to AI-driven change.

- Applied K-means clustering to 664 occupations across 128 skill features.
- Developed a Psychosocial Buffer Index to quantify protective job characteristics, including autonomy, task variety, and social interaction.
- Used principal component analysis (PCA), ANOVA, and Tukey HSD to examine skill structure and compare occupational clusters.

[Explore project](./occupational_ai_vulnerability/) · [Publication in the American Journal of Industrial Medicine](https://doi.org/10.1002/ajim.70131)

</details>

### 🤖 A simulated workplace, real questions

**Workplace Stress Simulation with Machine Learning and Reinforcement Learning**

Exploring stress, burnout, and interventions with agent-based simulation, ML, and RL.

<details>
<summary>🔎 Open the research notebook</summary>

Modelled workplace stress, burnout, absenteeism, and productivity using an agent-based simulation framework. Compared rule-based coping behaviour with a GRU-based worker action policy and a DQN-based organizational intervention policy across baseline, high-strain, and intervention scenarios.

[Explore project](./workplace_stress_abm_ml_rl/)

</details>

---

<a id="applied-lab"></a>

## 🛠️ Models with a mission

### 🛠️ Where can hospitals work better?

**Healthcare Resource Optimization**

Looking for unusual admission and readmission patterns, then comparing hospital workflows.

**My contribution:** Coded the model.

<details>
<summary>🔎 Open the research notebook</summary>

**University of Toronto · DataFest · May 2026**

- Developed a framework to flag underperforming hospitals based on unusually high admission and readmission patterns across common diagnoses.
- Proposed optimization strategies by comparing workflow differences, scheduling patterns, and follow-up frequencies across hospitals.

</details>

### 🛠️ Putting a price on risk

**Predictive Modelling for Renters' Insurance Pricing**

Feature engineering and XGBoost for renters' insurance pricing.

**My contribution:** Coded the model.

<details>
<summary>🔎 Open the research notebook</summary>

**Canadian Actuarial Society · Predictive Modeling Case Competition · January–March 2026**

- Developed XGBoost models to support renters' insurance pricing strategies and improve risk stratification.
- Engineered features and preprocessed structured insurance data to enhance predictive performance.

</details>

---

<a id="connection-lab"></a>

## 🤝 The science of connection

### 🤝 Small interactions, big questions

**Statistical Analysis of Social Isolation and Loneliness**

Exploring social interaction, isolation, loneliness, and mental wellness through survey data.

**My contribution:** Completed the coding, analysis, and poster creation.

<details>
<summary>🔎 Open the research notebook</summary>

**University of Toronto · Canadian Statistical Literacy Competition (ISLP National) · December 2024–March 2025**

- Analyzed Canadian Social Connection Survey data to study relationships between social interaction and loneliness.
- Applied regression modelling, permutation tests, and effect size analysis.

**Recognition:** Second place across Canada in the Canadian Statistical Literacy Competition, March 2025.

### Loneliness in the Elderly

Investigated social, demographic, and behavioural factors associated with loneliness among older adults using regression, exploratory analysis, and visualization.

[Explore project](./Loneliness_Elderly/)

### Social Connections and Mental Health

Investigated how social interactions, including greetings between strangers, relate to mental wellness using survey data, regression analysis, and visualization.

[Explore project](./CSCS_Project/)

</details>

---

## 🧰 My toolkit

- **Programming and computing:** Python, R, Bash, Linux, HPC, Git
- **Statistical methods:** Longitudinal analysis, mixed-effects models, simulation studies, regression, PCA
- **Machine learning:** scikit-learn, PyTorch, XGBoost
- **Neuroimaging and visualization:** FSL, NiBabel, Nilearn, PyRadiomics, Matplotlib

<div align="center">

**Thanks for wandering through my corner of GitHub ✨**

[Back to the top ↑](#hey-im-thinuri-)

</div>
