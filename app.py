from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import pandas as pd
import joblib
from datetime import date, timedelta
from collections import Counter
import warnings
import traceback

app = Flask(__name__)
app.secret_key = 'kunci_rahasia_anda_yang_sangat_aman'

# ==== Load model ====
def load_models():
    global rf_model, scaler, le, df, X
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rf_model = joblib.load("rf_weather_model.pkl")
            scaler   = joblib.load("scaler.pkl")
            le       = joblib.load("label_encoder.pkl")
        print("Model berhasil dimuat.")

        df = pd.read_csv("seattle-weather.csv")
        df = df.dropna(axis=1, how="all")
        X  = df.drop(columns=['weather'])
        X['date']  = pd.to_datetime(X['date'])
        X['month'] = X['date'].dt.month
        X['day']   = X['date'].dt.day
        X = X.drop(columns=['date'])

    except FileNotFoundError as e:
        print(f"File tidak ditemukan: {e}"); exit()
    except Exception as e:
        print(f"Error: {e}"); traceback.print_exc()
        try:
            rf_model = joblib.load("rf_weather_model.pkl", mmap_mode=None)
            scaler   = joblib.load("scaler.pkl",           mmap_mode=None)
            le       = joblib.load("label_encoder.pkl",    mmap_mode=None)
        except Exception as e2:
            print(f"Masih error: {e2}"); exit()

load_models()

users = {"admin": "admin123", "user": "user123"}

# ==== MULTI BAHASA ====
lang = {
    "id": {
        "app_name": "Sistem Pendukung Keputusan Pertanian",
        "dashboard": "Dashboard",
        "laporan_30_hari": "Laporan 30 Hari",
        "pengaturan": "Pengaturan",
        "logout": "Logout",
        "login": "Login",
        "username": "Username",
        "password": "Password",
        "login_btn": "Masuk",
        "login_error": "Username atau password salah!",
        "welcome": "Selamat datang",
        "cuaca_hari_ini": "Cuaca Hari Ini",
        "curah_hujan": "Curah Hujan",
        "suhu_maks": "Suhu Maks",
        "angin": "Angin",
        "rekomendasi_tanaman": "Rekomendasi Tanaman Bulan Ini",
        "peringatan_aktif": "Peringatan Aktif",
        "kegiatan_hari_ini": "Kegiatan Hari Ini",
        "lihat_laporan": "Lihat Laporan 30 Hari",
        "prediksi_tanggal": "Prediksi Tanggal Lain",
        "pilih_tanggal": "Pilih tanggal mulai",
        "proses": "Memproses prediksi 30 hari...",
        "empty_hint": "Pilih tanggal dan klik \"Lihat Laporan\" untuk mulai",
        "hari_cocok_tanam": "Hari cocok tanam",
        "rata_suhu": "Rata-rata suhu",
        "rata_curah_hujan": "Rata-rata curah hujan",
        "rata_angin": "Rata-rata angin",
        "data_harian": "Data Harian",
        "prediksi_per_tanggal": "Prediksi per tanggal — 30 hari",
        "tgl": "Tanggal",
        "curah_hujan_mm": "Curah Hujan (mm)",
        "suhu_maks_c": "Suhu Maks (°C)",
        "suhu_min_c": "Suhu Min (°C)",
        "angin_ms": "Angin (m/s)",
        "prediksi_cuaca": "Prediksi Cuaca",
        "cocok_tanam": "Cocok Tanam",
        "cocok": "Cocok",
        "hati_hati": "Hati-hati",
        "tidak_cocok": "Tidak cocok",
        "sangat_disarankan": "Sangat Disarankan",
        "kurang_disarankan": "Kurang Disarankan",
        "tidak_disarankan": "Tidak Disarankan",
        "disarankan_tanam": "Disarankan tanam",
        "kegiatan_disarankan": "Kegiatan Disarankan",
        "peringatan": "Peringatan",
        "tidak_ada_peringatan": "Tidak ada peringatan khusus",
        "tanaman": "Tanaman Terbaik",
        "skor": "Skor",
        "dari_30_hari": "dari 30 hari",
        "kondisi_baik": "Kondisi pertanian hari ini baik",
        "kondisi_perhatian": "Kondisi pertanian perlu perhatian",
        "hujan_tinggi": "Curah hujan tinggi — perhatikan kondisi lahan",
        "rendah": "Rendah",
        "sedang": "Sedang",
        "tinggi": "Tinggi",
        "normal": "Normal",
        "panas": "Panas",
        "dingin": "Dingin",
        "aman": "Aman",
        "kencang": "Kencang",
        "cukup": "Cukup",
        "baik": "Baik",
        "mm": "mm",
        "celsius": "°C",
        "ms": "m/s",
        "sun": "Cerah",
        "rain": "Hujan",
        "drizzle": "Gerimis",
        "fog": "Berkabut",
        "snow": "Salju",
        "tanaman_terbaik": "Tanaman Terbaik"
    },
    "en": {
        "app_name": "Agricultural Decision Support System",
        "dashboard": "Dashboard",
        "laporan_30_hari": "30-Day Report",
        "pengaturan": "Settings",
        "logout": "Logout",
        "login": "Login",
        "username": "Username",
        "password": "Password",
        "login_btn": "Sign In",
        "login_error": "Invalid username or password!",
        "welcome": "Welcome",
        "cuaca_hari_ini": "Today's Weather",
        "curah_hujan": "Precipitation",
        "suhu_maks": "Max Temperature",
        "angin": "Wind",
        "rekomendasi_tanaman": "Monthly Plant Recommendations",
        "peringatan_aktif": "Active Alerts",
        "kegiatan_hari_ini": "Today's Activities",
        "lihat_laporan": "View 30-Day Report",
        "prediksi_tanggal": "Predict Other Date",
        "pilih_tanggal": "Select start date",
        "proses": "Processing 30-day prediction...",
        "empty_hint": "Select a date and click \"View Report\" to start",
        "hari_cocok_tanam": "Good planting days",
        "rata_suhu": "Average temperature",
        "rata_curah_hujan": "Average precipitation",
        "rata_angin": "Average wind",
        "data_harian": "Daily Data",
        "prediksi_per_tanggal": "Daily prediction — 30 days",
        "tgl": "Date",
        "curah_hujan_mm": "Precipitation (mm)",
        "suhu_maks_c": "Max Temp (°C)",
        "suhu_min_c": "Min Temp (°C)",
        "angin_ms": "Wind (m/s)",
        "prediksi_cuaca": "Weather Prediction",
        "cocok_tanam": "Suitable for Planting",
        "cocok": "Suitable",
        "hati_hati": "Caution",
        "tidak_cocok": "Not Suitable",
        "sangat_disarankan": "Highly Recommended",
        "kurang_disarankan": "Less Recommended",
        "tidak_disarankan": "Not Recommended",
        "disarankan_tanam": "Recommended to plant",
        "kegiatan_disarankan": "Recommended Activities",
        "peringatan": "Warnings",
        "tidak_ada_peringatan": "No specific warnings",
        "tanaman": "Best Plant",
        "skor": "Score",
        "dari_30_hari": "of 30 days",
        "kondisi_baik": "Agricultural conditions are good today",
        "kondisi_perhatian": "Agricultural conditions need attention",
        "hujan_tinggi": "High precipitation — monitor field conditions",
        "rendah": "Low",
        "sedang": "Medium",
        "tinggi": "High",
        "normal": "Normal",
        "panas": "Hot",
        "dingin": "Cold",
        "aman": "Safe",
        "kencang": "Strong",
        "cukup": "Moderate",
        "baik": "Good",
        "mm": "mm",
        "celsius": "°C",
        "ms": "m/s",
        "sun": "Sunny",
        "rain": "Rainy",
        "drizzle": "Drizzle",
        "fog": "Foggy",
        "snow": "Snowy",
        
    }
}

def get_lang():
    """Ambil bahasa dari session, default Indonesia"""
    lang_code = session.get('lang', 'id')
    return lang.get(lang_code, lang['id'])

# ==== Prediksi 1 hari ====
def predict_weather(tanggal_input):
    try:
        date_obj = pd.to_datetime(tanggal_input)
    except:
        return {"error": "Format tanggal salah. Gunakan YYYY-MM-DD."}

    month, day = date_obj.month, date_obj.day
    subset = X[(X['month'] == month) & (X['day'] == day)]

    if subset.empty:
        precipitation = X['precipitation'].mean()
        temp_max      = X['temp_max'].mean()
        temp_min      = X['temp_min'].mean()
        wind          = X['wind'].mean()
    else:
        precipitation = subset['precipitation'].mean()
        temp_max      = subset['temp_max'].mean()
        temp_min      = subset['temp_min'].mean()
        wind          = subset['wind'].mean()

    new_data = pd.DataFrame({
        'precipitation': [precipitation], 'temp_max': [temp_max],
        'temp_min': [temp_min], 'wind': [wind],
        'month': [month], 'day': [day]
    })

    try:
        scaled = scaler.transform(new_data)
        pred   = rf_model.predict(scaled)
        label  = le.inverse_transform(pred)[0]
        return {
            "tanggal":        tanggal_input,
            "prediksi_cuaca": label,
            "precipitation":  round(float(precipitation), 2),
            "temp_max":       round(float(temp_max), 2),
            "temp_min":       round(float(temp_min), 2),
            "wind":           round(float(wind), 2)
        }
    except Exception as e:
        return {"error": f"Prediction error: {str(e)}"}


# ==== SPK ====
# Kebutuhan optimal tanaman dalam satuan NYATA
# precipitation (mm/hari), temp_max (°C), wind (m/s)
# cuaca_cocok: cuaca yang paling disukai tanaman
tanaman_data = {
    "Bayam":      {"precipitation": 8,  "temp_max": 25, "wind": 3, "cuaca_cocok": ["sun", "fog"]},
    "Tomat":      {"precipitation": 4,  "temp_max": 30, "wind": 2, "cuaca_cocok": ["sun"]},
    "Brokoli":    {"precipitation": 7,  "temp_max": 22, "wind": 3, "cuaca_cocok": ["sun", "fog"]},
    "Hidroponik": {"precipitation": 2,  "temp_max": 20, "wind": 1, "cuaca_cocok": ["sun", "fog", "drizzle"]},
    "Cabai":      {"precipitation": 4,  "temp_max": 32, "wind": 2, "cuaca_cocok": ["sun"]},
    "Jagung":     {"precipitation": 5,  "temp_max": 30, "wind": 2, "cuaca_cocok": ["sun"]},
    "Kentang":    {"precipitation": 6,  "temp_max": 25, "wind": 3, "cuaca_cocok": ["sun", "fog"]},
    "Ubi":        {"precipitation": 5,  "temp_max": 28, "wind": 3, "cuaca_cocok": ["sun", "fog"]},
}

# Bobot kriteria (total = 1.0)
bobot = {
    "precipitation":  0.30,
    "temp_max":       0.25,
    "wind":           0.15,
    "prediksi_cuaca": 0.30
}

# Batas maksimum normalisasi — SAMA untuk cuaca dan tanaman
NORM_MAX = {
    "precipitation": 10.0,  # mm/hari
    "temp_max":      25.0,  # °C
    "wind":          6.0   # m/s
}

# Mapping nilai cuaca aktual → skor 0–1
cuaca_mapping = {
    "sun":     1.0,
    "fog":     0.6,
    "drizzle": 0.5,
    "rain":    0.3,
    "snow":    0.1
}


def hitung_saw(weather):
    """
    SAW (Simple Additive Weighting) dengan kesesuaian:
      skor_i = Σ bobot_j × kesesuaian_ij

    Kesesuaian numerik  = 1 - |norm_tanaman - norm_cuaca|  (semua kriteria pakai rumus sama)
    Kesesuaian cuaca    = 1.0  jika cuaca masuk cuaca_cocok tanaman
                        = cuaca_mapping[label] × 0.4  jika tidak cocok tapi dikenal (gradasi)
                        = 0.0  jika tidak dikenal
    """
    # Normalisasi cuaca ke 0–1 (wind = cost, dibalik)
    norm_cuaca = {
        "precipitation": min(weather["precipitation"] / NORM_MAX["precipitation"], 1.0),
        "temp_max":      min(weather["temp_max"]      / NORM_MAX["temp_max"],      1.0),
        "wind":          1.0 - min(weather["wind"]    / NORM_MAX["wind"],          1.0),
    }
    cuaca_label = str(weather.get("prediksi_cuaca", "")).lower()

    hasil = []
    for nama, nilai in tanaman_data.items():
        # Normalisasi kebutuhan tanaman ke 0–1 (skala SAMA dengan cuaca)
        norm_tanaman = {
            "precipitation": min(nilai["precipitation"] / NORM_MAX["precipitation"], 1.0),
            "temp_max":      min(nilai["temp_max"]      / NORM_MAX["temp_max"],      1.0),
            "wind":          1.0 - min(nilai["wind"]    / NORM_MAX["wind"],          1.0),
        }

        # Kesesuaian numerik — rumus konsisten untuk semua kriteria
        kesesuaian = {
            k: 1.0 - abs(norm_tanaman[k] - norm_cuaca[k])
            for k in norm_cuaca
        }

        # Kesesuaian cuaca — 3 tier agar gradasi tetap bermakna
        if cuaca_label in nilai["cuaca_cocok"]:
            kesesuaian["prediksi_cuaca"] = 1.0                          # cocok penuh
        elif cuaca_label in cuaca_mapping:
            kesesuaian["prediksi_cuaca"] = cuaca_mapping[cuaca_label] * 0.4  # tidak cocok, tapi ada gradasi
        else:
            kesesuaian["prediksi_cuaca"] = 0.0                          # tidak dikenal

        # SAW: skor = Σ(bobot × kesesuaian)
        skor = sum(bobot[k] * kesesuaian[k] for k in bobot)

        hasil.append({
            "tanaman": nama,
            "skor":    round(skor, 4)
        })

    return sorted(hasil, key=lambda x: x["skor"], reverse=True)


# ==== Rekomendasi kegiatan (1 hari) ====
def rekomendasi_kegiatan(weather):
    """Untuk prediksi 1 hari (route /spk)."""
    hasil = []
    hujan = weather.get("precipitation", 0)
    angin = weather.get("wind", 0)
    cuaca = weather.get("prediksi_cuaca", "")

    if cuaca in ["rain", "drizzle", "snow"] or hujan > 10:
        hasil.append("❌ Tidak disarankan menanam (tanah terlalu basah)")
    elif cuaca == "fog":
        hasil.append("⚠️ Bisa menanam, perhatikan kelembaban")
    else:
        hasil.append("✅ Cocok untuk menanam")

    if hujan > 5:
        hasil.append("❌ Tidak perlu penyiraman (air sudah cukup)")
    elif hujan > 1:
        hasil.append("⚠️ Penyiraman ringan saja")
    else:
        hasil.append("✅ Perlu penyiraman")

    if angin > 7:
        hasil.append("❌ Hindari penyemprotan (angin kencang)")
    elif angin > 4:
        hasil.append("⚠️ Penyemprotan kurang efektif")
    else:
        hasil.append("✅ Aman untuk penyemprotan")

    if cuaca == "sun" and hujan < 1:
        hasil.append("🌞 Cocok untuk pemupukan")
    if cuaca in ["rain", "drizzle"]:
        hasil.append("⚠️ Waspada penyakit tanaman (jamur/bakteri)")

    return hasil


# ==== Rekomendasi kegiatan (30 hari) ====
def rekomendasi_kegiatan_30(daily):
    """Rekomendasi kegiatan berdasarkan data harian 30 hari."""
    hasil = []
    total = len(daily)

    hari_tanam   = [d for d in daily if d["prediksi_cuaca"] in ["sun", "fog"] and d["precipitation"] < 5]
    hari_siram   = [d for d in daily if d["precipitation"] < 1]
    hari_semprot = [d for d in daily if d["wind"] <= 4]
    hari_pupuk   = [d for d in daily if d["prediksi_cuaca"] == "sun" and d["precipitation"] < 1]
    hari_jamur   = [d for d in daily if d["prediksi_cuaca"] in ["rain", "drizzle"]]

    jml_tanam = len(hari_tanam)
    if jml_tanam == 0:
        hasil.append("❌ Tidak ada hari yang cocok untuk menanam dalam periode ini")
    elif jml_tanam < 7:
        hasil.append(f"⚠️ Hanya {jml_tanam} hari cocok menanam — pilih waktu dengan cermat")
    else:
        hasil.append(f"✅ {jml_tanam} dari {total} hari cocok untuk menanam")

    jml_siram = len(hari_siram)
    if jml_siram < 5:
        hasil.append("❌ Hampir tidak perlu penyiraman — curah hujan tinggi sepanjang periode")
    elif jml_siram < 15:
        hasil.append(f"⚠️ Penyiraman diperlukan sekitar {jml_siram} hari (saat hujan rendah)")
    else:
        hasil.append(f"✅ Penyiraman rutin disarankan ({jml_siram} hari curah hujan rendah)")

    jml_semprot = len(hari_semprot)
    if jml_semprot < 10:
        hasil.append(f"⚠️ Hanya {jml_semprot} hari aman untuk penyemprotan pestisida")
    else:
        hasil.append(f"✅ {jml_semprot} hari aman untuk penyemprotan pestisida")

    jml_pupuk = len(hari_pupuk)
    if jml_pupuk > 0:
        hasil.append(f"🌞 {jml_pupuk} hari cerah cocok untuk pemupukan")

    jml_jamur = len(hari_jamur)
    if jml_jamur > 15:
        hasil.append(f"⚠️ {jml_jamur} hari berpotensi jamur/bakteri — pantau kesehatan tanaman")
    elif jml_jamur > 5:
        hasil.append(f"⚠️ {jml_jamur} hari berpotensi penyakit tanaman — siapkan fungisida")

    return hasil


# ==== Peringatan ====
def peringatan(weather):
    hasil = []
    hujan = weather.get("precipitation", 0)
    angin = weather.get("wind", 0)
    cuaca = weather.get("prediksi_cuaca", "")

    if hujan > 30:
        hasil.append("🚨 Risiko banjir tinggi")
        hasil.append("⚠️ Potensi penyakit tanaman sangat tinggi")
    elif hujan > 10:
        hasil.append("⚠️ Potensi jamur dan bakteri meningkat")
    if cuaca in ["rain", "drizzle"]:
        hasil.append("⚠️ Kelembaban tinggi, rawan penyakit tanaman")
    if angin > 7:
        hasil.append("🚨 Angin kencang, risiko tanaman roboh")
    elif angin > 4:
        hasil.append("⚠️ Angin cukup kencang, hati-hati penyemprotan")

    return hasil


# ==== Prediksi 30 hari ====
def predict_30_days(start_date_str):
    try:
        start = pd.to_datetime(start_date_str)
    except:
        return {"error": "Format tanggal salah."}

    daily = []
    for i in range(30):
        d = start + timedelta(days=i)
        w = predict_weather(d.strftime('%Y-%m-%d'))
        if "error" not in w:
            daily.append(w)

    if not daily:
        return {"error": "Tidak ada data prediksi."}

    # Agregat
    avg_precip   = round(sum(d["precipitation"] for d in daily) / len(daily), 2)
    avg_temp     = round(sum(d["temp_max"]      for d in daily) / len(daily), 2)
    avg_wind     = round(sum(d["wind"]          for d in daily) / len(daily), 2)
    total_precip = round(sum(d["precipitation"] for d in daily), 2)

    good_days = sum(
        1 for d in daily
        if d["prediksi_cuaca"] in ["sun", "fog"] and d["precipitation"] < 5
    )

    # Cuaca dominan
    cuaca_count    = Counter(d["prediksi_cuaca"] for d in daily)
    dominant_cuaca = cuaca_count.most_common(1)[0][0]

    # SPK rata-rata 30 hari
    avg_weather = {
        "precipitation":  avg_precip,
        "temp_max":       avg_temp,
        "wind":           avg_wind,
        "prediksi_cuaca": dominant_cuaca
    }
    ranking  = hitung_saw(avg_weather)
    kegiatan = rekomendasi_kegiatan_30(daily)
    warning  = peringatan(avg_weather)

    # Tandai tanaman yang layak ditanam (semua kondisi harus terpenuhi)
    for i, item in enumerate(ranking):
        item["bisa_panen"] = (
            item["skor"] >= 0.65          # skor kesesuaian cukup tinggi
            and good_days >= 12           # minimal 12 hari cuaca baik dalam 30 hari
            and avg_precip <= 8.0         # curah hujan rata-rata tidak terlalu tinggi
            and dominant_cuaca != "rain"  # cuaca dominan bukan hujan
        )
        # Kategori tampilan: 3 primadona, alternatif, tidak disarankan
        if i < 3:
            item["kategori"] = "primadona"
        elif item["skor"] >= 0.50:
            item["kategori"] = "alternatif"
        else:
            item["kategori"] = "tidak_disarankan"

    return {
        "start_date":            start_date_str,
        "end_date":              (start + timedelta(days=29)).strftime('%Y-%m-%d'),
        "avg_precipitation":     avg_precip,
        "total_precipitation":   total_precip,
        "avg_temp_max":          avg_temp,
        "avg_wind":              avg_wind,
        "good_days":             good_days,
        "dominant_cuaca":        dominant_cuaca,
        "daily":                 daily,
        "ranking_tanaman":       ranking,
        "kegiatan":              kegiatan,
        "peringatan":            warning
    }


# ==== Routes ====
@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/set-language/<lang_code>')
def set_language(lang_code):
    if lang_code in ['id', 'en']:
        session['lang'] = lang_code
    # Redirect ke halaman sebelumnya, atau dashboard jika tidak ada referrer
    ref = request.referrer
    if ref:
        return redirect(ref)
    if session.get('username') == 'admin':
        return redirect(url_for('admin_dashboard'))
    return redirect(url_for('user_dashboard'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'logged_in' in session:
        return redirect(url_for('user_dashboard'))
    
    if request.method == 'POST':
        u = request.form['username']
        p = request.form['password']
        if u in users and users[u] == p:
            session['logged_in'] = True
            session['username']  = u
            session['lang'] = session.get('lang', 'id')  # default bahasa
            return redirect(url_for('user_dashboard'))
        return render_template('login.html', error='Username atau password salah!', lang=get_lang())
    return render_template('login.html', lang=get_lang())

@app.route('/dashboard')
def user_dashboard():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    return render_template('index.html', username=session['username'], lang=get_lang())

@app.route('/prediksi-harian')
def prediksi_harian():
    if 'logged_in' in session and session['username'] == 'user':
        return render_template('spk.html', username=session['username'], lang=get_lang())
    return redirect(url_for('login'))

@app.route('/laporan-page')
def laporan_page():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    return render_template('laporan.html', username=session['username'], lang=get_lang())

@app.route('/admin')
def admin_dashboard():
    if 'logged_in' not in session or session.get('username') != 'admin':
        return redirect(url_for('login'))
    try:
        weather_data = pd.read_csv("seattle-weather.csv").to_dict('records')
        return render_template('admin_page.html', username=session['username'], data=weather_data, features=list(X.columns), lang=get_lang())
    except Exception as e:
        return f"Error: File admin_page.html tidak ditemukan atau rusak. Detail: {e}"

@app.route('/history')
def history():
    if 'logged_in' in session and session['username'] == 'admin':
        history_data = [
            {'tanggal': '2024-01-15', 'prediksi': 'Cerah',   'akurasi': '85%'},
            {'tanggal': '2024-01-16', 'prediksi': 'Hujan',   'akurasi': '78%'},
            {'tanggal': '2024-01-17', 'prediksi': 'Berawan', 'akurasi': '82%'},
            {'tanggal': '2024-01-18', 'prediksi': 'Cerah',   'akurasi': '88%'},
            {'tanggal': '2024-01-19', 'prediksi': 'Hujan',   'akurasi': '75%'},
        ]
        return render_template('history.html', username=session['username'], history_data=history_data, lang=get_lang())
    return redirect(url_for('login'))

@app.route('/settings')
def settings():
    if 'logged_in' in session and session['username'] == 'admin':
        try:
            system_info = {
                'model_version': '1.0.0',
                'accuracy':      '89%',
                'last_trained':  '2024-01-15',
                'data_points':   len(df),
                'features':      list(X.columns)
            }
            return render_template('settings.html', username=session['username'], system_info=system_info, lang=get_lang())
        except Exception as e:
            return f"Error: {e}"
    return redirect(url_for('login'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/predict', methods=['POST'])
def predict():
    if 'logged_in' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    data = request.get_json()
    return jsonify(predict_weather(data['date']))

@app.route('/spk', methods=['POST'])
def spk():
    if 'logged_in' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    data    = request.get_json()
    weather = predict_weather(data['date'])
    if "error" in weather:
        return jsonify(weather)
    ranking  = hitung_saw(weather)
    kegiatan = rekomendasi_kegiatan(weather)
    warning  = peringatan(weather)
    return jsonify({**weather, "ranking_tanaman": ranking, "kegiatan": kegiatan, "peringatan": warning})

@app.route('/laporan', methods=['POST'])
def laporan():
    if 'logged_in' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    data = request.get_json()
    return jsonify(predict_30_days(data['date']))

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

# ==== API PENGATURAN ====

@app.route('/pengaturan')
def pengaturan_page():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    if session['username'] != 'admin':
        return redirect(url_for('user_dashboard'))
    return render_template('pengaturan.html', 
                         username=session['username'],
                         tanaman_data=tanaman_data,
                         bobot=bobot,
                         cuaca_mapping=cuaca_mapping,
                         norm_max=NORM_MAX,
                         lang=get_lang())

@app.route('/api/tanaman', methods=['GET', 'POST', 'DELETE'])
def api_tanaman():
    if session.get('username') != 'admin':
        return jsonify({"error": "Akses ditolak"}), 403
    
    if request.method == 'GET':
        return jsonify(tanaman_data)
    
    if request.method == 'POST':
        data = request.get_json()
        nama = data.get('nama')
        if not nama:
            return jsonify({"error": "Nama tanaman diperlukan"}), 400
        if nama in tanaman_data:
            return jsonify({"error": "Tanaman sudah ada"}), 400
        
        tanaman_data[nama] = {
            "precipitation": float(data.get('precipitation', 0)),
            "temp_max": float(data.get('temp_max', 25)),
            "wind": float(data.get('wind', 2)),
            "cuaca_cocok": data.get('cuaca_cocok', ['sun'])
        }
        return jsonify({"message": f"Tanaman {nama} berhasil ditambahkan", "data": tanaman_data[nama]})
    
    if request.method == 'DELETE':
        data = request.get_json()
        nama = data.get('nama')
        if not nama:
            return jsonify({"error": "Nama tanaman diperlukan"}), 400
        if nama not in tanaman_data:
            return jsonify({"error": "Tanaman tidak ditemukan"}), 404
        
        del tanaman_data[nama]
        return jsonify({"message": f"Tanaman {nama} berhasil dihapus"})

@app.route('/api/bobot', methods=['GET', 'POST'])
def api_bobot():
    if 'logged_in' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    if session.get('username') != 'admin':
        return jsonify({"error": "Akses ditolak"}), 403
    
    if request.method == 'GET':
        return jsonify({"bobot": bobot, "norm_max": NORM_MAX, "cuaca_mapping": cuaca_mapping})
    
    if request.method == 'POST':
        data = request.get_json()
        new_bobot = data.get('bobot', {})
        
        total = sum(float(v) for v in new_bobot.values())
        if abs(total - 1.0) > 0.001:
            return jsonify({"error": f"Total bobot harus 1.0 (sekarang {total:.2f})"}), 400
        
        for k in bobot:
            if k in new_bobot:
                bobot[k] = float(new_bobot[k])
        
        if 'norm_max' in data:
            norm_data = data['norm_max']
            for k in NORM_MAX:
                if k in norm_data:
                    NORM_MAX[k] = float(norm_data[k])
        
        return jsonify({"message": "Bobot berhasil diperbarui", "bobot": bobot, "norm_max": NORM_MAX})

if __name__ == '__main__':
    app.run(debug=True)



