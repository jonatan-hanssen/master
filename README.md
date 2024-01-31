# Explainable AI for Data Outlier Detection

Denne READMEen er foreløpig hovedsaklig overordnede notater


## Overordnet

Ha et datasett med kjente out of distribution punkter, kjør XAI OOD detector og state of the art OOD detectors og se om man får bedre resultater.

## Datasett

- Det trengs både in distribution (ID) og out-of-distribution (OOD) data
- Ideelt sett hadde man f.eks hatt ett datasett fra ett sykehus og ett fra ett annet
    - Eller et datasett hvor man kunne tatt ut deler av det for å lage noe som er OOD, f.eks tatt ut alle av en viss alder
- HyperKvasir er anonymisert data fra to sykehus, med bruk av to ulike endoskop
    - Man kan definitivt ikke hente ut grupper av personer, da alt er anonymisert
    - Uklart om man kan skille endoskopene, sannsynligvis ikke
- Mulighet: ta annet datasett som da kan være OOD

## XAI

- Grad-CAM
    - Brukes ofte i medisin
- Meaningful perturbation
- Hausdorff distance

## Out of distribution detection

- Covariate shift, ikke semantic shift. Ingen nye klasser.
- State of the art: VOS, ViM

## Ideer

- Bare kjøre outlier detection på resultatet av Grad-CAM?

## Lesestoff

### Lest

- Selvaraju et Al. [*Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*](https://arxiv.org/pdf/1610.02391.pdf). (2019).
- Zhou et Al. [*Learning Deep Features for Discriminative Localization*](https://arxiv.org/pdf/1512.04150.pdf). (2015).
- Yang et Al. [*Generalized Out-of-Distribution Detection: A Survey*](https://arxiv.org/pdf/2110.11334.pdf). (2021).
- Velden et Al. [*Explainable artificial intelligence (XAI) in deep learning-based medical image analysis*](https://www.sciencedirect.com/science/article/pii/S1361841522001177#bib0252). (2022).
- Molnar, Cristoph. *Interpretable Machine Learning*. (2023).
- Borgli et Al. [*HyperKvasir, a comprehensive multi-class image and video dataset for gastrointestinal endoscopy*](https://www.nature.com/articles/s41597-020-00622-y). (2020).
- Hägele et Al. [*Resolving challenges in deep learning-based analyses of histopathological images using explanation methods*](https://www.nature.com/articles/s41598-020-62724-2.pdf). (2020).

### Ikke lest

- Wickstrøm et Al. [*Uncertainty and interpretability in convolutional neural networks for semantic segmentation of colorectal polyps*](https://www.sciencedirect.com/science/article/pii/S1361841519301574). (2019).
- Wang et Al. [*ViM: Out-Of-Distribution with Virtual-logit Matching*](https://arxiv.org/pdf/2203.10807.pdf). (2022).
- Du et Al. [*VOS: Learning What You Don’t Know By Virtual Outlier synthesis*](https://arxiv.org/pdf/2202.01197.pdf). (2020).
- Nazir et Al. [*Survey of explainable artificial intelligence techniques for biomedical imaging with deep neural networks*](https://www.sciencedirect.com/science/article/pii/S0010482523001336?ref=cra_js_challenge&fr=RR-1). (2023).
- Liu et Al. [*Energy-based Out-of-distribution Detection*](https://arxiv.org/pdf/2010.03759.pdf). (2021).
- Cui et Al. [*Out-of-Distribution (OOD) Detection Based on Deep Learning: A Review](https://www.mdpi.com/2079-9292/11/21/3500). (2022).
