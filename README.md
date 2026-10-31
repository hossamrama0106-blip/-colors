# كاشف منتجات خيام

موقع ثابت (GitHub Pages) يتعرف على المنتج من صورة (لصق / رفع / سحب / كاميرا) ويعرض اسم المنتج من khiam.store ونسبة التطابق.
المطابقة تتم داخل المتصفح بنموذج CLIP (`Xenova/clip-vit-base-patch32`) مقابل فهرس جاهز لصور المتجر وصور الأرشيف.

## النشر
1. ارفع محتويات هذا المجلد (بدون `tools/work`) إلى مستودع GitHub.
2. Settings → Pages → Branch: `main` / root.

## تحديث البيانات (عند تغيّر المنتجات أو إضافة صور)
```
cd tools
python fetch_catalog.py     # يسحب المنتجات والصور من khiam.store
python export_catalog.py
python build_index.py       # يبني الفهرس (ببطء — ~25 دقيقة لـ 3900 صورة)
python link_archive.py      # يربط صور الأرشيف بالمنتجات المتاحة حاليًا
```
متطلبات: `pip install onnxruntime huggingface_hub pillow numpy requests`.

## منطق النتيجة
- منتج على الموقع + تشابه ≥ 0.90: **تطابق شبه مؤكد**؛ ≥ 0.70: **مقارب**؛ ≥ 0.50: **ضعيف**؛ أقل: **لم يتم التأكد**.
- لو الأقرب صورة من الأرشيف غير مرتبطة بأي منتج حالي: **منتج قديم نفذ** + أقرب منتج متاح.
- العتبات في `data/calib.json`.
