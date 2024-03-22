# 1. Dataset

**Name:** [Wizard of Wikipedia](https://parl.ai/projects/wizard_of_wikipedia/) ([Download](http://parl.ai/downloads/wizard_of_wikipedia/wizard_of_wikipedia.tgz))

**Domain:** single (Wikipedia)

**Modality:** textual

**Collection method:** in-house employing human knowledge experts (wizards)

**DS (component) type:** NLU & DM & NLG

**Annotation:** relevant topics and passages from wiki pages, chosen dialog topic, chosen dialog passages, wizard score

**Format:** .json

**License:** free

# 2. Measures

**Total data length:**

- **user:**
    - dialogs: 9275
    - turns: 83540
    - sentences: 153300
    - words: 1223957
- **system:**
    - dialogs: 9155
    - turns: 83247
    - sentences: 171244
    - words: 1568737

**Mean/std dev dialogue lengths:**

- **user:**
    - turns: 4.532555615843733 / 0.539199048594081
    - sentences: 8.317145957677699 / 3.223200544882356
    - words: 66.40651112316874 / 27.590230445442526
- **system:**
    - turns: 4.516657623440043 / 0.5238045056188101 
    - sentences: 9.290992946283234 / 3.374168492188647
    - words: 85.1113944655453 / 29.299042613197006

**Vocabulary size:**

- **user:** 31559
- **system:** 44809

**Shannon entropy over words:**

- **user:** 9.567900091891431
- **system:** 10.306886503515141

# 3. Analysis

**Does the data look natural?** Yes, system dialog turns looks pretty natural since wizards use emotional expressions

**How difficult do you think this dataset will be to learn from?** There is enough data for reverse engineering (learning from) considering the full annotation of topics and passages corresponding to the dialog turns

**How usable will it be in an actual system?** This type of data would be useful in development of a wizard for wiki-like systems that consists of multiple information pages

**Do you think there's some kind of problem or limitation with the data?** Not every topic is covered in wiki articles (for example, "how to boil eggs?" or "cite my favourite niche song by some unpopular artist"), thus there are domain limitation indeed. 