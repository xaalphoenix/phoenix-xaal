# PHOENIX STL Studio

آماده‌سازی فایل STL برای پرینت سه‌بعدی: **برش**، **اتصال** و **تعمیر**، حتی برای فایل‌های خیلی سنگین (۷۰۰MB و بیشتر).
ورودی و خروجی فقط STL است. فایل خروجی مستقیم در Elegoo SatelLite، Chitubox و Lychee باز می‌شود.

> وضعیت: **فاز ۲ در حال ساخت**. مرحله‌ی A (جابه‌جایی، Undo، Merge)، B (برش‌های پیشرفته) و C (پین و زبانه) آماده‌اند. نقشه‌ی کامل پروژه در [PLAN.md](PLAN.md) است.

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

## قابلیت‌های فاز ۲ (تا اینجا)

- **جابه‌جایی، چرخش و Scale** قطعه‌ها با موس (کشیدن روی میز، Shift برای بالا و پایین) یا عدد. «روی میز»، «وسط میز» و **خواباندن روی یک سطح**.
- **Undo / Redo** (تا ۳۰ قدم) برای همه‌ی کارها.
- انتخاب چندتایی، **Merge** (یکی کردن جامد) و **Combine** (کنار هم).
- پروفایل پرینتر: Saturn 4 Ultra 16K، FDM عمومی و پروفایل‌های دلخواه.
- **برش Grid:** چند صفحه در X/Y/Z با هم؛ هر صفحه جدا قابل جابه‌جایی.
- **جا دادن در پرینتر:** کمترین تعداد تکه که هر کدام در پرینتر جا شوند (چرخش ۹۰ درجه هم امتحان می‌شود).
- **برش با خط منحنی:** روی مدل نقطه بگذارید (Shift برای گوشه‌ی تیز)؛ خط در جهت دید از مدل رد می‌شود.
- **برش با سطح آزاد:** صفحه‌ای با نقطه‌های کنترلی که با موس خم می‌شود.
- **فاصله‌ی اتصال (Joint gap):** برای برش‌های منحنی و آزاد، تا تکه‌ها بعد از پرینت راحت روی هم بنشینند.

### اتصال تکه‌ها (تب «اتصال»)

- دو تکه‌ی بریده‌شده را انتخاب کنید: سطح مشترکشان خودکار پیدا می‌شود. برای یک تکه، «انتخاب سطح صاف» و کلیک روی سطح.
- انواع اتصال: **پین جدا** (پین‌ها به‌صورت یک قطعه‌ی جدا ساخته می‌شوند)، **پین چسبیده**، **جای آهن‌ربا** (اندازه‌های رایج ۳×۱ تا ۱۰×۳)، **جای میله‌ی فلزی**، **خار چهارگوش** (ضد چرخش)، **زبانه و شیار** دور تا دور، **دم‌چلچله**ی کشویی.
- **لقی دقیق:** فاصله از هر طرف بر اساس نوع پرینتر (رزین/FDM) و نوع جفت شدن (پرسی، جذب، کشویی، شل)، به‌اضافه‌ی عمق اضافه‌ی سوراخ، پخ، و حداقل دیواره. سوراخ‌ها هیچ‌وقت از اندازه‌ی خواسته‌شده تنگ‌تر نمی‌شوند.
- **جای‌گذاری خودکار** با رعایت حداقل دیواره در تمام عمق سوراخ، یا دستی (کلیک برای اضافه کردن، کشیدن برای جابه‌جایی). نقطه‌ی قرمز یعنی دیواره کافی نیست.
- **مدل‌های توخالی:** جایی که دیواره نازک است، یک ستون توپر (Boss) داخل مدل اضافه می‌شود که هم‌شکل دیواره است و از سطح بیرونی بیرون نمی‌زند. زبانه خودکار وسط دیواره‌ی نازک قرار می‌گیرد.
- **قطعه‌ی تست لقی:** صفحه‌ای با ۵ سوراخ و یک پین. بهترین سوراخ را انتخاب کنید تا لقی برای همان پرینتر ذخیره شود.
- اگر سوراخی از دیواره بیرون بزند، بعد از اعمال گزارش می‌شود (Undo کار می‌کند).

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
| `Ctrl+Z` / `Ctrl+Y` | برگشت / انجام دوباره |
| `Backspace` / `Enter` | حذف آخرین نقطه / پایان کشیدن خط (برش منحنی) |

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
  core/      numpy mesh engine: io_stl, weld, analyze, repair, cut, grid, surface_cut, boolean,
             transform, section, connectors, lod, hardware
  engine/    separate process that owns full-resolution parts (on disk); progress, cancel, crash isolation
  ui/        PySide6 + pyvistaqt: main window, viewport, parts, panels, hover help, i18n
  i18n/      en/fa UI strings and hover-help texts (JSON)
```

The UI only holds light preview meshes. Every heavy operation runs in the engine process on the full-resolution data, so the window never freezes, and Cancel kills the job immediately.
