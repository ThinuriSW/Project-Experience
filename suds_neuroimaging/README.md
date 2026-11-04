[← Back to all projects](../README.md)

# 🧠 Mapping how brains change

**Longitudinal MRI-Based Modelling of Neurodegenerative Disease Progression**

Following brain trajectories over time to understand neurodegenerative disease.

## 📂 Explore the work

| Material | Open |
| :--- | :--- |
| Research poster | [View poster · PNG](https://github.com/user-attachments/assets/e7973ebb-e827-4408-a3be-9745f62d480d) |
| Research report | [Read report · PDF](https://github.com/user-attachments/files/34382657/SUDS_Report.pdf) |
| Trajectory maps | [View figure · PNG](https://github.com/user-attachments/assets/8998b9fc-fba0-4bb5-9e85-abba3505f20d) |
| Acceleration classes | [View figure · PNG](https://github.com/user-attachments/assets/ecd367d4-8773-4298-b129-a15da602acab) |

## 🖼️ Poster

[![SUDS research poster](https://github.com/user-attachments/assets/e7973ebb-e827-4408-a3be-9745f62d480d)](https://github.com/user-attachments/assets/e7973ebb-e827-4408-a3be-9745f62d480d)

**The project at a glance.** My SUDS poster brings together the motivation, MRI preprocessing, nonlinear trajectory model, and example brain maps. Click the poster to open it at full size.

[📄 Read the full SUDS report](https://github.com/user-attachments/files/34382657/SUDS_Report.pdf)

## 🧠 From a fitted curve to a changing brain map

[![SUDS report Figure 9 showing grey-matter density, velocity, and acceleration trajectories beside brain maps at frames 0, 30, 60, and 90](https://github.com/user-attachments/assets/8998b9fc-fba0-4bb5-9e85-abba3505f20d)](https://github.com/user-attachments/assets/8998b9fc-fba0-4bb5-9e85-abba3505f20d)

**Figure 9 · What changes, how fast, and whether that change is speeding up.** Fitted trajectories for an example symptomatic bvFTD participant sit alongside sagittal slices from the 4D maps. Rows show grey-matter density, velocity, and acceleration; columns show normalized frames 0, 30, 60, and 90. Negative velocity indicates decline. During decline, negative acceleration means faster loss, while positive acceleration means slowing loss.

## 🎨 Mapping patterns of acceleration

[![SUDS report Figure 10 showing voxelwise acceleration classes in sagittal, coronal, and axial brain views, with a categorical legend](https://github.com/user-attachments/assets/ecd367d4-8773-4298-b129-a15da602acab)](https://github.com/user-attachments/assets/ecd367d4-8773-4298-b129-a15da602acab)

**Figure 10 · Different trajectory patterns across the brain.** The colored regions group voxels by the signs of acceleration and its rate of change. Group 0 is unassigned; groups 1–4 are the trajectory classes defined in the report. The thin green crosshairs mark the selected slice location, rather than a class.

*These are exploratory trajectory visualizations, not validated conversion predictions. Full methods, uncertainty calculations, and limitations are in the [report](https://github.com/user-attachments/files/34382657/SUDS_Report.pdf).*


## 🔬 Methods and research context

**Baycrest Academy for Research · Wang Lab, Rotman Research Institute · May 2026–Present**  
SUDS Scholar (May–August 2026), continuing as a Research Volunteer (September 2026–Present). Co-supervised by Dr. Lei Wang and Dr. Yann Cobigo.

- Developed voxelwise Bayesian mixed-effects models in logit space, mapping fitted trajectories back to grey-matter space to study behavioural variant frontotemporal dementia (bvFTD).
- Generated brain maps showing the rate of tissue loss and how it changes over time.
- Used Linux-based high-performance computing clusters for computationally intensive MRI analyses.
- Developing an interactive FSLeyes plugin to inspect individual grey-matter density, velocity, and acceleration trajectories alongside brain maps.

**Research output:** Generated voxelwise 4D maps of grey-matter density, velocity, and acceleration to explore how the location and rate of tissue change evolve over time.

**Current stage:** Exploratory disease-trajectory modelling. Conversion prediction and quantitative comparison with a linear baseline remain to be validated.

Presented at the SUDS Research Showcase, University of Toronto, in August 2026.
