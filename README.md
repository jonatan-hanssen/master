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
