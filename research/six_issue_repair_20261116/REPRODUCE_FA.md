# گام‌های بازتولید بسته ۶ ایراد

نیازمندی‌ها: Python 3.11+، `numpy`, `scipy`, `matplotlib`، و فایل قبلی موجود در مخزن `research/anytime_mixture_20261116/anytime_mixture.py` (با گواهی Beta-mixture مستقل).

از ریشهٔ مخزن GitHub اجرا کنید:

```bash
python -m pip install numpy scipy matplotlib
cd research/six_issue_repair_20261116
python -m unittest -v test_six_issues
python baselines.py 40
python drift_cert.py 40
python scaling.py
```

**احتیاط:** `baselines.py` در صورت اجرای مستقیم، جدول ۴۰بذری بدون چرخه را ایجاد می‌کند. برای چرخه حقیقی:

```bash
python -c "from baselines import benchmark; import json; print(json.dumps({k:v for k,v in benchmark(40,cyclic=True).items() if k!='rows'}, indent=2))"
```

گزارش جدول و نمودارهای منابع قبل از ادعای تمام‌شدن نوآوری مستقل باید با اجرای CI روی SHA نهایی مطابقت داده شوند.