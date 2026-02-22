---
title: Lit Ner Tagger
emoji: 🐢
colorFrom: green
colorTo: red
sdk: gradio
sdk_version: 6.5.1
app_file: app.py
pinned: false
short_description: Tagger for MNER of literary texts
---

## Named Entity Recognition for Literary Texts

A Gradio web app for detecting named entities in literary texts in **English**, **French**, and **Italian**.

Built on a fine-tuned [XLM-RoBERTa](https://huggingface.co/WpnSta/lner-xlm-roberta) model trained on literary texts from the 19th and 20th centuries.

### Entity Types
| Tag | Meaning |
|---|---|
| PER | Person, character, or animal with an active narrative role |
| GPE | Geo-political entity (e.g. London, the city) |
| LOC | Location (e.g. the sea, the beach) |
| ORG | Organization (e.g. the army, the court) |
| FAC | Facility (e.g. the kitchen, the street) |
| VEH | Vehicle (e.g. the coach, the boats) |
| TIME | Temporal or historical reference (e.g. in the morning) |

### Live Demo
Try it on [Hugging Face Spaces](https://huggingface.co/spaces/WpnSta/lit_ner_tagger).
