# Weight-Space Discovery of an Epistatic Circuit Invisible to Activation-Based Methods

Code, data, and figures for the RTI (Repeated Token Identification) circuit paper.

## The circuit

15 heads in GPT-2 Small organized into four tiers:

| Tier | Heads | Function |
|------|-------|----------|
| Backbone | L0H8, L0H9, L0H11 | Positional encoding |
| Detector | L4H11 | Previous-token head |
| Copier | L4H0, L5H6, L5H7, L7H0, L8H4, L8H7, L9H3, L9H10 | Attend unique, boost logits |
| Readout | L10H11, L11H9, L11H11 | Induction + amplification |

The eight-head copier tier is unrecoverable by five tested activation-based methods (ACDC, EAP, EAP-IG, activation patching, copy scores). 63% of the circuit's causal effect arises from interactions between heads.

## Repository structure

```
paper/              Paper source (tex, bib, TMLR style files)
figures/            Discovery process figures (heatmaps, clustering, validation)
data/
  paper_numbers/    All experimental results (JSON)
  prereg/           Pre-registration documents
scripts/            Experiment scripts (path patching, classifiers, etc.)
lab_notebooks/      Discovery chronicle and distilled lab notebook
```

## Citation

```bibtex
@article{tower2026rti,
  title={Weight-Space Discovery of an Epistatic Circuit Invisible to Activation-Based Methods},
  author={Tower, Elliot},
  year={2026},
}
```
