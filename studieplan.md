# Explainable Artificial Intelligence for data outlier detection - Studieplan ROBIN 2023-2025 
### Jonatan Hoffmann Hanssen

---------------

### Prosjektbeskrivelse

Prosjektet går ut på utforske om Explainable AI (XAI) kan brukes til å avdekke out-of-domain datapunkter. XAI gir oss en begrunnelse for hvorfor en maskinlæringsmodell har kommet til en spesifikk beslutning. Tanken er dermed at man kan bruke denne informasjonen til å oppdage suspekte beslutninger, f.eks. dersom begrunnelsen på et datapunkt avviker veldig fra begrunnelsen i de fleste andre, eller dersom begrunnelsen viser at modellen legger vekt på deler av datapunktet som vi intuitivt vet ikke burde ha noe å si for prediksjonen (f.eks dersom en modell fokuserer på bakgrunnen i et bilde framfor objektet som klassifiseres). Å kunne avdekke slike datapunkter er viktig, da en modell kan gi tilsynelatende høy konfidens på et slikt datapunkt, slik at man vil ha vanskelig for å skille gyldige beslutninger fra de ugyldige.

Prosjektet vil fokusere på medisinsk bildedata, mer spesifikt [HyperKvasir](https://datasets.simula.no/hyper-kvasir/). Dette er et datasett med 110 079 gastrointestinale bilder, hvorav 10 662 har blitt annotert. I medisinske applikasjoner er både forklarbarhet og usikkerhetsmåling viktige forskingsområder, og det er derfor et bredt spekter av bakgrunnslitteratur å sette seg inn i. Prosjektet vil dermed gå ut på å utvikle en metode som bruker XAI til å oppdage når en prediksjon på dette datasettet er blitt gjort på feilaktig grunnlag, fordi punktet er out-of-domain. Dette vil sannsynligvis bli gjort ved å se på visuelle forklaringer, som *saliency maps*. Andre XAI metoder som kan gi nyttig informasjon er eksempelbaserte forklaringer, hvor nettverket gir lignende datapunkter for å begrunne sitt svar, eller *SHAP*.

### Fremdriftsplan

#### Emner

| Emne | Semester
| --- | ---
| IN5490 | Høst 2023
| TEK5040 | Høst 2023
| TEK5020 | Høst 2023
| STK4900 | Vår 2024
| IN4310 | Vår 2024
| IN5310 | Høst 2024

I vår 2024 vil jeg begynne å sette meg inn i bakgrunnslitteratur og se på state-of-the-art innenfor XAI, med hovedfokus på metoder som kan brukes på bildeklassifisering. Jeg vil ta emnet IN4310 for å få en bedre forståelse for dyp læring for bildeanalyse, som vil gi meg grunnlag for å forstå hvordan XAI kan brukes på konvolusjonelle nettverk. Jeg vil også ta emnet STK4900 for å få bedre kunnskap om statistiske metoder som kan brukes for å modellere usikkerhet og som kan være nyttig sammen med XAI for å oppdage out-of-domain punkter. Jeg vil også skrive essay. I høsten 2024 vil jeg begynne å utforske ulike metoder for å oppdage out-of-domain datapunkter via XAI, og forhåpentligvis finne gode kandidater. Muligens kan jeg velge én av disse å gå videre med. Jeg vil også ta IN5310 for å få ytterligere fordypning i dyp læring for bildeanalyse. I våren 2025 vil jeg forhåpentligvis ha gode resultater og kan bruke tiden på å skrive og finpusse på modellen.


## Lesestoff

- Molnar, Cristoph. *Interpretable Machine Learning*. (2023).
- Velden et Al. [*Explainable artificial intelligence (XAI) in deep learning-based medical image analysis*](https://www.sciencedirect.com/science/article/pii/S1361841522001177#bib0252). (2022).
- Wickstrøm et Al. [*Uncertainty and interpretability in convolutional neural networks for semantic segmentation of colorectal polyps*](https://www.sciencedirect.com/science/article/pii/S1361841519301574). (2019).
- Itoh et Al. [*Visualising decision-reasoning regions in computer-aided pathological pattern diagnosis of endoscytoscopic images based on CNN weights analysis*](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/11314/2549532/Visualising-decision-reasoning-regions-in-computer-aided-pathological-pattern-diagnosis/10.1117/12.2549532.short?SSO=1#_=_). (2020).
- Hägele et Al. [*Resolving challenges in deep learning-based analyses of histopathological images using explanation methods*](https://www.nature.com/articles/s41598-020-62724-2.pdf). (2020).
- Wang et Al. [*ViM: Out-Of-Distribution with Virtual-logit Matching*](https://arxiv.org/pdf/2203.10807.pdf). (2022).
- Du et Al. [*VOS: Learning What You Don’t Know By Virtual Outlier synthesis*](https://arxiv.org/pdf/2202.01197.pdf). (2020).
- Yang et Al. [*Generalized Out-of-Distribution Detection: A Survey*](https://arxiv.org/pdf/2110.11334.pdf). (2021).
- Borgli et Al. [*HyperKvasir, a comprehensive multi-class image and video dataset for gastrointestinal endoscopy*](https://www.nature.com/articles/s41597-020-00622-y). (2020).
