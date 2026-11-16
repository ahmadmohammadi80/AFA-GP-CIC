# Real pretrained language-model pilot

This code loads **HuggingFaceTB/SmolLM2-135M-Instruct** on a CPU worker and computes reference probabilities over six answer candidates for 12 multiple-choice prompts. It then solves a KL-constrained full-policy certificate problem.

**IMPORTANT SCIENTIFIC LIMITATIONS:** The preference feedback is **programmatic and simulated from correctness labels**, including an artificial cycle on incorrect answers; it is *not* observed human preference. This is a real-model integration pilot, not an RLHF benchmark, a claim of real-world preference alignment, or an inference-speed comparison to Best-of-N.

The baseline policy is a *tempered, smoothed conditional distribution on six completions*, not the unmodified model output distribution.

Run:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install 'transformers>=4.47,<5' numpy scipy huggingface_hub
python research/real_llm_pilot.py --out outputs/real_llm.json
```

The GitHub Actions workflow `.github/workflows/afa-real-llm.yml` downloads the public model and uploads `outputs/real_llm.json` as an artifact. To establish real-human alignment, use an independent human preference benchmark and multiple models, with comparisons matched in budget and task difficulty.
