# Hi, I'm Thinuri Welithotage

I'm a University of Toronto student specializing in Mathematics and Its Applications (Statistics and Probability). My research and projects span longitudinal modelling, machine learning, neuroimaging, occupational health, and social connection.

This portfolio brings together my research contributions, statistical analyses, and applied modelling projects. Links below lead to project materials currently available in this repository; projects without a materials link are summarized here.

## Longitudinal MRI-Based Modelling of Neurodegenerative Disease Progression

**Baycrest Academy for Research · Wang Lab, Rotman Research Institute · May 2026–Present**  
SUDS Scholar (May–August 2026), continuing as a Research Volunteer (September 2026–Present). Co-supervised by Dr. Lei Wang and Dr. Yann Cobigo.

- Developed voxelwise models of longitudinal MRI data to map gray-matter change in behavioural variant frontotemporal dementia (bvFTD).
- Generated brain maps showing the rate of tissue loss and how it changes over time.
- Used Linux-based high-performance computing clusters for computationally intensive MRI analyses.
- Built an interactive FSLeyes tool to visualize individual brain trajectories.

Presented at the SUDS Research Showcase, University of Toronto, in August 2026.

## Bias in Linear Mixed Models with Outcome-Dependent Follow-up

**University of Toronto · January–April 2026**  
Supervised by Dr. Omidali (Omid) Aghababaei Jazi.

- Developed a simulation framework in which observation times depend on prior outcomes.
- Modelled longitudinal data using linear mixed-effects models with random intercepts and slopes.
- Evaluated estimation bias, mean squared error, and confidence interval coverage when standard ignorability assumptions are violated.

## Simulation and Machine Learning for Longitudinal Workplace Survey Data

**University of Toronto · ReSTORE Lab · February 2026–Present**

- Developed a reproducible synthetic dataset representing workplace conditions, stress-related measures, and survey participation across 1,000 simulated workers over 90 days.
- Evaluated GRU sequence models alongside logistic regression and personal response-history baselines to predict next-day survey nonresponse.
- Assessed performance using worker-separated evaluation, additional split and training runs, calibration diagnostics, and checks for information leakage.

## Occupational Vulnerability to AI-Driven Change

**University of Toronto · ReSTORE Lab · September 2025–February 2026**

Examined how skill composition, task structure, and psychosocial buffers shape occupational vulnerability to AI-driven change.

- Applied K-means clustering to 664 occupations across 128 skill features.
- Developed a Psychosocial Buffer Index to quantify protective job characteristics, including autonomy, task variety, and social interaction.
- Used principal component analysis (PCA), ANOVA, and Tukey HSD to examine skill structure and compare occupational clusters.

[Explore project](./occupational_ai_vulnerability/) · [Publication in the American Journal of Industrial Medicine](https://doi.org/10.1002/ajim.70131)

## Healthcare Resource Optimization

**University of Toronto · DataFest · May 2026**

- Developed a framework to flag underperforming hospitals based on unusually high admission and readmission patterns across common diagnoses.
- Proposed optimization strategies by comparing workflow differences, scheduling patterns, and follow-up frequencies across hospitals.

## Predictive Modelling for Renters' Insurance Pricing

**Canadian Actuarial Society · Predictive Modeling Case Competition · January–March 2026**

- Developed XGBoost models to support renters' insurance pricing strategies and improve risk stratification.
- Engineered features and preprocessed structured insurance data to enhance predictive performance.

## Alzheimer's Biomarker Detection Using MRI

**STEM Fellowship · National Inter-University Health Data & AI Program · May–August 2025**

- Built machine learning models using scikit-learn and PyTorch to investigate early-stage Alzheimer's biomarkers from MRI data.
- Extracted radiomic features from OASIS-2 structural and longitudinal MRI data using PyRadiomics.
- Processed neuroimaging data using NiBabel, Nilearn, and SimpleITK.
- Evaluated model performance using cross-validation and feature selection.

Presented through the 2025 STEM Fellowship program and at the University of Toronto's Arts and Science Union Undergraduate Research Conference in March 2026.

[Explore project](./Alzheimer_Biomarkers/) · [Watch presentation](https://www.youtube.com/watch?v=W4WdSREN0iU) · [Conference paper](https://doi.org/10.48448/pby5-a658)

## Statistical Analysis of Social Isolation and Loneliness

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

## Workplace Stress Simulation with Machine Learning and Reinforcement Learning

Modelled workplace stress, burnout, absenteeism, and productivity using an agent-based simulation framework. Compared rule-based coping behaviour with a GRU-based worker action policy and a DQN-based organizational intervention policy across baseline, high-strain, and intervention scenarios.

[Explore project](./workplace_stress_abm_ml_rl/)

## Technical Skills

- **Programming and computing:** Python, R, Bash, Linux, HPC, Git
- **Statistical methods:** Longitudinal analysis, mixed-effects models, simulation studies, regression, PCA
- **Machine learning:** scikit-learn, PyTorch, XGBoost
- **Neuroimaging and visualization:** FSL, NiBabel, Nilearn, PyRadiomics, Matplotlib
