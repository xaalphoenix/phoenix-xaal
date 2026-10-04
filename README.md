# PHOENIX STL Studio

آماده‌سازی فایل STL برای پرینت سه‌بعدی: **برش**، **اتصال** و **تعمیر**، حتی برای فایل‌های خیلی سنگین (۷۰۰MB و بیشتر).
ورودی و خروجی فقط STL است. فایل خروجی مستقیم در Elegoo SatelLite، Chitubox و Lychee باز می‌شود.

> وضعیت: **فاز ۱ (پروتوتایپ)**. نقشه‌ی کامل پروژه در [PLAN.md](PLAN.md) است.

## اجرا روی ویندوز (بدون نصب Python)

1. در GitHub به تب **Actions** بروید و آخرین اجرای موفق **Windows build** را باز کنید.
2. پایین صفحه، از بخش **Artifacts** فایل `PhoenixSTLStudio-win64` را دانلود کنید.
3. فایل zip را باز کنید و `PhoenixSTLStudio\PhoenixSTLStudio.exe` را اجرا کنید.
   اگر ویندوز هشدار SmartScreen داد: **More info ← Run anyway** (برنامه هنوز امضای دیجیتال ندارد).

## قابلیت‌های فاز ۱

- باز کردن STL سنگین بدون هنگ. پردازش در یک پروسه‌ی جداگانه انجام می‌شود و دکمه‌ی «لغو» همیشه کار می‌کند.
- **برش صفحه‌ای** با دستگیره‌ی سه‌بعدی یا عدد (جهت، دو چرخش، موقعیت). خط برش زنده نمایش داده می‌شود و هر دو قطعه‌ی خروجی جامد بسته و قابل پرینت‌اند.
- **برش مرحله‌به‌مرحله:** هر قطعه را دوباره برش بزنید. همه‌ی قطعه‌ها در لیست «قطعه‌ها» می‌مانند.
- **بررسی و تعمیر خودکار:** سوراخ، سطح برعکس، مثلث خراب و تکراری، تکه‌های شناور. دیواره‌ی داخلی مدل‌های توخالی سالم می‌ماند.
- نمایش حجم پرینتر (Saturn 4 Ultra 16K یا FDM عمومی) و اینکه قطعه جا می‌شود یا نه.
- رابط **فارسی و انگلیسی**. موس که روی هر تنظیمی بماند، `⋯` و راهنمای آن تنظیم ظاهر می‌شود (یا F1).
- خروجی STL باینری (میلی‌متر)، هر قطعه یک فایل.

## میانبرها

| کلید | کار |
|---|---|
| `Ctrl+O` | باز کردن STL (یا فایل را روی پنجره بکشید) |
| `Ctrl+E` / `Ctrl+Shift+E` | خروجی قطعه‌ی انتخاب‌شده / همه‌ی قطعه‌های نمایان |
| `Ctrl+I` / `Ctrl+R` | بررسی / تعمیر خودکار |
| `Home` | نمای پیش‌فرض |
| `Ctrl+1/2/3` | نمای بالا / روبه‌رو / کنار |
| `F1` | راهنمای تنظیمی که موس روی آن است |
| `Del` | حذف قطعه |

---

## English

Prepare STL files for 3D printing: **cut**, **connect** and **repair**, even very heavy files. STL in, STL out.

**Run on Windows:** open the latest successful **Windows build** run under the repository's *Actions* tab, download the `PhoenixSTLStudio-win64` artifact, unzip it, and run `PhoenixSTLStudio.exe`.

**Run from source** (Python 3.10+):

```bash
pip install -e ".[dev]"
phoenix-stl model.stl --lang en      # or: python -m phoenix_stl
```

**Develop:**

```bash
pytest tests/test_core.py tests/test_engine.py        # core + engine process
xvfb-run -a pytest tests/test_ui.py                   # UI smoke test (Linux)
python tools/make_heavy_stl.py big.stl --triangles 15000000
python tools/bench.py big.stl                          # timings + peak memory per step
pyinstaller packaging/phoenix_stl.spec                 # frozen build
```

### Architecture

```
src/phoenix_stl/
  core/      numpy mesh engine: io_stl, weld, analyze, repair, cut, lod, hardware
  engine/    separate process that owns full-resolution parts (on disk); progress, cancel, crash isolation
  ui/        PySide6 + pyvistaqt: main window, viewport, parts, panels, hover help, i18n
  i18n/      en/fa UI strings and hover-help texts (JSON)
```

The UI only holds light preview meshes. Every heavy operation runs in the engine process on the full-resolution data, so the window never freezes, and Cancel kills the job immediately.
