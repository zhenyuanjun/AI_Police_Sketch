# Final Interpretation: Age Model and AGE-6

## Photo-domain performance

On the AgeDB test set, the model achieved an FP32 MAE of 9.69 years and an INT8 MAE of 9.72 years. These photo-domain results describe performance on AgeDB photographs; they do not establish corresponding accuracy on generated sketches.

## Optimization

INT8 retained nearly identical AgeDB test MAE (+0.31% relative change) while reducing measured backbone invoke-only latency by 75.47% (4.08× mean speedup) and model size by 74.16%. Latency is batch size 1 and excludes the full web application.

## Generated-sketch experiment

Across the 12 formal AGE-6 trials, witness age and base ML Estimated Age showed strong ordering (Pearson r = 0.963; Spearman ρ = 0.946). Absolute calibration showed compression toward middle ages: 20-year sketches had a mean base estimate of 31.843 years, while 60-year sketches had a mean of 45.340 years.

## Refinement response

Younger/older edits moved ML Estimated Age in the requested direction in 10 of 12 trials (83.3%): 5/6 younger trials and 5/6 older trials. Mean change was −8.11 years for younger edits and +9.86 years for older edits. Failures remain included in the formal results.

## Limitation

The model provides an exploratory perceptual signal for age-related visual consistency in generated sketches. This experiment does not establish chronological age accuracy on sketches, forensic validity, identity verification, demographic fairness, or transfer of AgeDB MAE to police sketches.
