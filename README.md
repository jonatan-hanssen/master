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

## Lesestoff

### Lest

- Selvaraju et Al. [*Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*](https://arxiv.org/pdf/1610.02391.pdf). (2019).
- Zhou et Al. [*Learning Deep Features for Discriminative Localization*](https://arxiv.org/pdf/1512.04150.pdf). (2015).
- Yang et Al. [*Generalized Out-of-Distribution Detection: A Survey*](https://arxiv.org/pdf/2110.11334.pdf). (2021).
- Velden et Al. [*Explainable artificial intelligence (XAI) in deep learning-based medical image analysis*](https://www.sciencedirect.com/science/article/pii/S1361841522001177#bib0252). (2022).
- Molnar, Cristoph. *Interpretable Machine Learning*. (2023).
- Borgli et Al. [*HyperKvasir, a comprehensive multi-class image and video dataset for gastrointestinal endoscopy*](https://www.nature.com/articles/s41597-020-00622-y). (2020).

### Ikke lest

- Wickstrøm et Al. [*Uncertainty and interpretability in convolutional neural networks for semantic segmentation of colorectal polyps*](https://www.sciencedirect.com/science/article/pii/S1361841519301574). (2019).
- Itoh et Al. [*Visualising decision-reasoning regions in computer-aided pathological pattern diagnosis of endoscytoscopic images based on CNN weights analysis*](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/11314/2549532/Visualising-decision-reasoning-regions-in-computer-aided-pathological-pattern-diagnosis/10.1117/12.2549532.short?SSO=1#_=_). (2020).
- Hägele et Al. [*Resolving challenges in deep learning-based analyses of histopathological images using explanation methods*](https://www.nature.com/articles/s41598-020-62724-2.pdf). (2020).
- Wang et Al. [*ViM: Out-Of-Distribution with Virtual-logit Matching*](https://arxiv.org/pdf/2203.10807.pdf). (2022).
- Du et Al. [*VOS: Learning What You Don’t Know By Virtual Outlier synthesis*](https://arxiv.org/pdf/2202.01197.pdf). (2020).

