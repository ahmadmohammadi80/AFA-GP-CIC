# بسته تجربی بازتولیدپذیر AFA-GP-CIC — ۱۸ نوامبر ۲۰۲۶

این بسته دارای کد Python قابل اجرا، ۹ آزمون واحد، نتایج خام چهار مجموعه، گزارش فارسی و ابزار بازبینی نتایج است.

## پیش‌نیازها

- Python 3.13 و کتابخانه‌های فهرست‌شده در `requirements.txt`. برای سایر نسخه‌های Python سازگاری مستقل راستی‌آزمایی نشده است.
- اجرا در محیط محلی؛ اتصال اینترنت برای اجرای خود آزمایش لازم نیست، مشروط بر نصب بسته‌ها.

## اجرای سریع

```bash
python -m pip install -r requirements.txt
python verify_results.py
cd research/six_issues_20261116
python -m unittest -v test_integrated_two_step_certificate_20261118 test_online_two_step_benchmark_20261118
```

## بازاجرای آزمایش‌های کامل

از پوشه ریشه بسته، در لینوکس/macOS:

```bash
bash reproduce.sh
```

یا دستورات کامل را از `docs/ONLINE_EXPERIMENT_COMPLETE_20261118_FA.md` اجرا کنید. فایل‌های خام پیشین در `results/` باقی می‌مانند؛ اجرای جدید در `results/rerun/` ذخیره می‌شود. برای بررسی فایل‌های جدید:

```bash
python verify_results.py --rerun
```

## خلاصه اجراهای ثبت‌شده

| مجموعه | greedy | joint_mix | AFA gated 2-step |
|---|---:|---:|---:|
| اصلی، ۴۰ بذر | ۳۴ | ۳۸ | ۳۴ |
| مستقل، ۴۰ بذر | ۳۴ | ۳۷ | ۳۴ |
| چرخه غیرانتقالی، ۲۰ بذر | ۱۸ | ۲۰ | ۱۸ |
| null، ۲۰ بذر | ۰ | ۰ | ۰ |

**دقت:** این بسته ادعای برتری عمومیِ آنلاین یا آزمون تجربی مدل زبانی واقعی ندارد. نسخه آنلاین از فعال‌ساز نزدیک آستانه برای برنامه‌ریز دقیق استفاده می‌کند. بررسی نوآوری سازوکار مستقل در سند `docs/COMPETITOR_MECHANISM_AUDIT_20261118_FA.md` قرار دارد و باید جدا از آزمون مزیت عملکردی تعبیر شود.

## منشأ کد

- مخزن خصوصی: `ahmadmohammadi80/AFA-GP-CIC`، شاخه `research/complete-six-issue-package-20261116`.
- فایل دقیق قدیمی `integrated_two_step_certificate_20261118.py` از commit `b7eec416bd947ef011063916612401f03e912a77` برگرفته شده است.
- تابع confidence/certificate از مخزن (`research/anytime_mixture_20261116/anytime_mixture.py`؛ blob SHA `3b4e8d2b58938344f2aa0f66a6238507e663475d`) به‌صورت آینه حداقلی سه تابع استفاده‌شده در این آزمایش قرار گرفته است.
- فایل‌های `online_two_step_benchmark_20261118.py`، `test_online_two_step_benchmark_20261118.py` و گزارش کامل در این بسته تولید و در محیط محلی آزموده شدند. هر زمان push تأیید نشود نباید برایشان SHA جدید GitHub ادعا شود.
- manifest فایل‌ها در `SHA256SUMS.txt` است و باید قبل از اعتماد به بسته بررسی شود.