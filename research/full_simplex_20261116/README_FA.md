# بسته آزمایش و اثبات AFA-GP-CIC

**سه بخش پژوهشی:** (۱) تمایز سازوکار بهینه‌سازی گواهی روی تمام سیمپلکس KL؛ (۲) قضیه جداسازی در مقابل هر رقیب محدود به یک انتقال جرم؛ (۳) آزمایش تصادفی چرخه‌ای و طرح ارزیابی با مدل پیش‌آموزش‌دیده.

## اجرا در CPU

```bash
pip install numpy scipy
python strong_adaptive_study.py --ks 6 12 24 48 --n 5000 --seeds 40 --out cyclic_study.json
python -m unittest discover -s tests -v
python real_llm_pilot.py --dry-run --n 3000 --out dry_run.json
```

**توجه:** `--dry-run` از توزیع‌های مصنوعی استفاده می‌کند و ارزیابی LLM واقعی نیست. برای LLM واقعی:

```bash
pip install 'transformers>=4.47,<5' torch huggingface-hub
python real_llm_pilot.py --model HuggingFaceTB/SmolLM2-135M-Instruct --out outputs/real_llm.json
```

در آزمایش واقعی، مدل صرفاً *احتمالات مرجع* شش گزینه را می‌دهد؛ رأی‌های ترجیحی توسط یک سنجش‌گر ساختگی/برنامه‌نویسی‌شده بر پایه پاسخ درست تولید می‌شوند. این یک آزمایش پایلوت واقعیِ مدل با **داور مصنوعی** است، نه مقایسه انسانی و نه آزمایش RLHF واقعی. برای ادعای کارایی هم‌ترازی انسانی، داده/داور مستقل ترجیحی لازم است.

**واژه‌های کلیدی:** cyclic preferences, KL, robust certificate, direct baseline opponent, single-transfer oracle, unrestricted KL oracle.