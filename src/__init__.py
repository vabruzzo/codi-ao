# CODI Activation Oracle
#
# Modules:
#   config.py              - All configuration dataclasses
#   codi_loader.py         - Load CODI model + checkpoint
#   activation_extractor.py - Extract per-thought activations from CODI
#   cot_parser.py          - Parse GSM8k-Aug CoT into structured steps
#   thought_alignment.py   - Align thoughts to CoT steps
#   qa_templates.py        - Question template paraphrases (6 categories)
#   qa_generator.py        - Generate QA pairs from activations
#   ao_dataset.py          - Build TrainingDataPoint objects for AO
#   steering.py            - Norm-matched activation injection hook
#   ao_trainer.py          - AO training loop
#   ao_eval.py             - Evaluation suite
#   utils.py               - Shared helpers
