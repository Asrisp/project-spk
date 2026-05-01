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
        "app_name": "Kota Seattle, Amerika",
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
        "tanaman_terbaik": "Tanaman Terbaik",
        "tidak_disarankan_menanam": "Tidak disarankan menanam (tanah terlalu basah)",
        "bisa_menanam_perhatikan": "Bisa menanam, perhatikan kelembaban",
        "cocok_untuk_menanam": "Cocok untuk menanam",
        "tidak_perlu_penyiraman": "Tidak perlu penyiraman (air sudah cukup)",
        "penyiraman_ringan": "Penyiraman ringan saja",
        "perlu_penyiraman": "Perlu penyiraman",
        "hindari_penyemprotan": "Hindari penyemprotan (angin kencang)",
        "penyemprotan_kurang_efektif": "Penyemprotan kurang efektif",
        "aman_penyemprotan": "Aman untuk penyemprotan",
        "cocok_pemupukan": "Cocok untuk pemupukan",
        "waspada_penyakit": "Waspada penyakit tanaman (jamur/bakteri)",
        "tidak_ada_hari_cocok": "Tidak ada hari yang cocok untuk menanam dalam periode ini",
        "hari_cocok_sedikit": "hanya {} hari cocok menanam — pilih waktu dengan cermat",
        "hari_cocok_banyak": "{} dari {} hari cocok untuk menanam",
        "hampir_tidak_perlu_siram": "Hampir tidak perlu penyiraman — curah hujan tinggi sepanjang periode",
        "penyiraman_diperlukan": "Penyiraman diperlukan sekitar {} hari (saat hujan rendah)",
        "penyiraman_rutin": "Penyiraman rutin disarankan ({} hari curah hujan rendah)",
        "hari_aman_semprot_sedikit": "Hanya {} hari aman untuk penyemprotan pestisida",
        "hari_aman_semprot": "{} hari aman untuk penyemprotan pestisida",
        "hari_cerah_pupuk": "{} hari cerah cocok untuk pemupukan",
        "potensi_jamur_tinggi": "{} hari berpotensi jamur/bakteri — pantau kesehatan tanaman",
        "potensi_penyakit": "{} hari berpotensi penyakit tanaman — siapkan fungisida",
        "risiko_banjir": "Risiko banjir tinggi",
        "potensi_penyakit_tinggi": "Potensi penyakit tanaman sangat tinggi",
        "potensi_jamur_meningkat": "Potensi jamur dan bakteri meningkat",
        "kelembaban_tinggi": "Kelembaban tinggi, rawan penyakit tanaman",
        "angin_kencang_roboh": "Angin kencang, risiko tanaman roboh",
        "angin_cukup_kencang": "Angin cukup kencang, hati-hati penyemprotan",
        "high_humidity_disease": "Kelembaban tinggi, rawan penyakit tanaman",
        "tanaman_bayam": "Bayam",
        "tanaman_tomat": "Tomat",
        "tanaman_brokoli": "Brokoli",
        "tanaman_hidroponik": "Hidroponik",
        "tanaman_cabai": "Cabai",
        "tanaman_jagung": "Jagung",
        "tanaman_kentang": "Kentang",
        "tanaman_ubi": "Ubi",
        "kelola_tanaman": "Kelola Tanaman",
        "ubah_bobot": "Ubah Bobot",
        "tambah_tanaman_baru": "Tambah Tanaman Baru",
        "nama_tanaman": "Nama Tanaman",
        "contoh_tanaman": "Contoh: Selada",
        "curah_hujan_ideal": "Curah Hujan Ideal (mm/hari)",
        "suhu_maks_ideal": "Suhu Maks Ideal (°C)",
        "angin_maks": "Angin Maks (m/s)",
        "cuaca_cocok_label": "Cuaca yang Cocok",
        "tambah_tanaman": "Tambah Tanaman",
        "daftar_tanaman": "Daftar Tanaman",
        "bobot_kriteria": "Bobot Kriteria SAW",
        "batas_normalisasi": "Batas Normalisasi Maksimum",
        "max_precipitation": "Max Precipitation (mm)",
        "max_temp": "Max Temp (°C)",
        "max_wind": "Max Wind (m/s)",
        "simpan_bobot": "Simpan Bobot & Normalisasi",
        "gagal_muat": "Gagal memuat data",
        "belum_ada_tanaman": "Belum ada tanaman",
        "hapus": "Hapus",
        "nama_tanaman_harus_diisi": "Nama tanaman harus diisi",
        "pilih_minimal_1_cuaca": "Pilih minimal 1 cuaca",
        "gagal_menambah": "Gagal menambah tanaman",
        "gagal_menghapus": "Gagal menghapus tanaman",
        "gagal_menyimpan": "Gagal menyimpan bobot",
        "total_bobot_harus": "Total bobot harus 1.00",
        "sekarang": "sekarang",
        "konfirmasi_hapus": "Hapus tanaman \"{nama}\"?",
        "total_label": "Total",
        "hujan_short": "Hujan",
        "suhu_short": "Suhu",
        "angin_short": "Angin",
        "lupa_password": "Lupa password?",
    },
    "en": {
        "app_name": "Seattle City, USA",
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
        "tanaman_terbaik": "Best Plant",
        "tidak_disarankan_menanam": "Not recommended to plant (soil too wet)",
        "bisa_menanam_perhatikan": "Can plant, watch humidity levels",
        "cocok_untuk_menanam": "Suitable for planting",
        "tidak_perlu_penyiraman": "No watering needed (sufficient rain)",
        "penyiraman_ringan": "Light watering only",
        "perlu_penyiraman": "Watering needed",
        "hindari_penyemprotan": "Avoid spraying (strong winds)",
        "penyemprotan_kurang_efektif": "Spraying less effective",
        "aman_penyemprotan": "Safe for spraying",
        "cocok_pemupukan": "Suitable for fertilization",
        "waspada_penyakit": "Watch for plant diseases (fungus/bacteria)",
        "tidak_ada_hari_cocok": "No suitable planting days in this period",
        "hari_cocok_sedikit": "only {} suitable planting days — choose timing carefully",
        "hari_cocok_banyak": "{} out of {} days suitable for planting",
        "hampir_tidak_perlu_siram": "Almost no watering needed — high rainfall throughout",
        "penyiraman_diperlukan": "Watering needed about {} days (when rainfall is low)",
        "penyiraman_rutin": "Regular watering recommended ({} days with low rainfall)",
        "hari_aman_semprot_sedikit": "Only {} days safe for pesticide spraying",
        "hari_aman_semprot": "{} days safe for pesticide spraying",
        "hari_cerah_pupuk": "{} sunny days suitable for fertilization",
        "potensi_jamur_tinggi": "{} days with potential fungus/bacteria — monitor plant health",
        "potensi_penyakit": "{} days with potential plant disease — prepare fungicide",
        "risiko_banjir": "High flood risk",
        "potensi_penyakit_tinggi": "Very high potential for plant disease",
        "potensi_jamur_meningkat": "Increased potential for fungus and bacteria",
        "kelembaban_tinggi": "High humidity, prone to plant disease",
        "angin_kencang_roboh": "Strong winds, risk of plants falling over",
        "angin_cukup_kencang": "Strong winds, be careful with spraying",
        "high_humidity_disease": "High humidity, prone to plant disease",
        "tanaman_bayam": "Spinach",
        "tanaman_tomat": "Tomato",
        "tanaman_brokoli": "Broccoli",
        "tanaman_hidroponik": "Hydroponics",
        "tanaman_cabai": "Chili Pepper",
        "tanaman_jagung": "Corn",
        "tanaman_kentang": "Potato",
        "tanaman_ubi": "Sweet Potato",
        "kelola_tanaman": "Manage Plants",
        "ubah_bobot": "Change Weights",
        "tambah_tanaman_baru": "Add New Plant",
        "nama_tanaman": "Plant Name",
        "contoh_tanaman": "Example: Lettuce",
        "curah_hujan_ideal": "Ideal Precipitation (mm/day)",
        "suhu_maks_ideal": "Ideal Max Temp (°C)",
        "angin_maks": "Max Wind (m/s)",
        "cuaca_cocok_label": "Suitable Weather",
        "tambah_tanaman": "Add Plant",
        "daftar_tanaman": "Plant List",
        "bobot_kriteria": "SAW Criteria Weights",
        "batas_normalisasi": "Max Normalization Limits",
        "max_precipitation": "Max Precipitation (mm)",
        "max_temp": "Max Temp (°C)",
        "max_wind": "Max Wind (m/s)",
        "simpan_bobot": "Save Weights & Normalization",
        "gagal_muat": "Failed to load data",
        "belum_ada_tanaman": "No plants yet",
        "hapus": "Delete",
        "nama_tanaman_harus_diisi": "Plant name is required",
        "pilih_minimal_1_cuaca": "Select at least 1 weather",
        "gagal_menambah": "Failed to add plant",
        "gagal_menghapus": "Failed to delete plant",
        "gagal_menyimpan": "Failed to save weights",
        "total_bobot_harus": "Total weight must be 1.00",
        "sekarang": "now",
        "konfirmasi_hapus": "Delete plant \"{nama}\"?",
        "total_label": "Total",
        "hujan_short": "Rain",
        "suhu_short": "Temp",
        "angin_short": "Wind",
        "lupa_password": "Forgot password?",
    }
}

def get_lang():
    """Ambil bahasa dari session, default Indonesia"""
    lang_code = session.get('lang', 'id')
    return lang.get(lang_code, lang['id'])

def translate_tanaman(nama):
    """Terjemahkan nama tanaman berdasarkan session bahasa"""
    lg = get_lang()
    key = f"tanaman_{nama.lower()}"
    return lg.get(key, nama)

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
tanaman_data = {
    "bayam":      {"precipitation": 8,  "temp_max": 25, "wind": 3, "cuaca_cocok": ["sun", "fog"]},
    "tomat":      {"precipitation": 4,  "temp_max": 30, "wind": 2, "cuaca_cocok": ["sun"]},
    "brokoli":    {"precipitation": 7,  "temp_max": 22, "wind": 3, "cuaca_cocok": ["sun", "fog"]},
    "hidroponik": {"precipitation": 2,  "temp_max": 20, "wind": 1, "cuaca_cocok": ["sun", "fog", "drizzle"]},
    "cabai":      {"precipitation": 4,  "temp_max": 32, "wind": 2, "cuaca_cocok": ["sun"]},
    "jagung":     {"precipitation": 5,  "temp_max": 30, "wind": 2, "cuaca_cocok": ["sun"]},
    "kentang":    {"precipitation": 6,  "temp_max": 25, "wind": 3, "cuaca_cocok": ["sun", "fog"]},
    "ubi":        {"precipitation": 5,  "temp_max": 28, "wind": 3, "cuaca_cocok": ["sun", "fog"]},
}

bobot = {
    "precipitation":  0.30,
    "temp_max":       0.25,
    "wind":           0.15,
    "prediksi_cuaca": 0.30
}

NORM_MAX = {
    "precipitation": 10.0,
    "temp_max":      25.0,
    "wind":          6.0
}

cuaca_mapping = {
    "sun":     1.0,
    "fog":     0.6,
    "drizzle": 0.5,
    "rain":    0.3,
    "snow":    0.1
}


def hitung_saw(weather):
    norm_cuaca = {
        "precipitation": min(weather["precipitation"] / NORM_MAX["precipitation"], 1.0),
        "temp_max":      min(weather["temp_max"]      / NORM_MAX["temp_max"],      1.0),
        "wind":          1.0 - min(weather["wind"]    / NORM_MAX["wind"],          1.0),
    }
    cuaca_label = str(weather.get("prediksi_cuaca", "")).lower()

    hasil = []
    for nama, nilai in tanaman_data.items():
        norm_tanaman = {
            "precipitation": min(nilai["precipitation"] / NORM_MAX["precipitation"], 1.0),
            "temp_max":      min(nilai["temp_max"]      / NORM_MAX["temp_max"],      1.0),
            "wind":          1.0 - min(nilai["wind"]    / NORM_MAX["wind"],          1.0),
        }

        kesesuaian = {
            k: 1.0 - abs(norm_tanaman[k] - norm_cuaca[k])
            for k in norm_cuaca
        }

        if cuaca_label in nilai["cuaca_cocok"]:
            kesesuaian["prediksi_cuaca"] = 1.0
        elif cuaca_label in cuaca_mapping:
            kesesuaian["prediksi_cuaca"] = cuaca_mapping[cuaca_label] * 0.4
        else:
            kesesuaian["prediksi_cuaca"] = 0.0

        skor = sum(bobot[k] * kesesuaian[k] for k in bobot)

        hasil.append({
            "tanaman": translate_tanaman(nama),
            "skor":    round(skor, 4)
        })

    return sorted(hasil, key=lambda x: x["skor"], reverse=True)


# ==== Rekomendasi kegiatan (1 hari) ====
def rekomendasi_kegiatan(weather):
    lg = get_lang()
    hasil = []
    hujan = weather.get("precipitation", 0)
    angin = weather.get("wind", 0)
    cuaca = weather.get("prediksi_cuaca", "")

    if cuaca in ["rain", "drizzle", "snow"] or hujan > 10:
        hasil.append("❌ " + lg["tidak_disarankan_menanam"])
    elif cuaca == "fog":
        hasil.append("⚠️ " + lg["bisa_menanam_perhatikan"])
    else:
        hasil.append("✅ " + lg["cocok_untuk_menanam"])

    if hujan > 5:
        hasil.append("❌ " + lg["tidak_perlu_penyiraman"])
    elif hujan > 1:
        hasil.append("⚠️ " + lg["penyiraman_ringan"])
    else:
        hasil.append("✅ " + lg["perlu_penyiraman"])

    if angin > 7:
        hasil.append("❌ " + lg["hindari_penyemprotan"])
    elif angin > 4:
        hasil.append("⚠️ " + lg["penyemprotan_kurang_efektif"])
    else:
        hasil.append("✅ " + lg["aman_penyemprotan"])

    if cuaca == "sun" and hujan < 1:
        hasil.append("🌞 " + lg["cocok_pemupukan"])
    if cuaca in ["rain", "drizzle"]:
        hasil.append("⚠️ " + lg["waspada_penyakit"])

    return hasil


# ==== Rekomendasi kegiatan (30 hari) ====
def rekomendasi_kegiatan_30(daily):
    lg = get_lang()
    hasil = []
    total = len(daily)

    hari_tanam   = [d for d in daily if d["prediksi_cuaca"] in ["sun", "fog"] and d["precipitation"] < 5]
    hari_siram   = [d for d in daily if d["precipitation"] < 1]
    hari_semprot = [d for d in daily if d["wind"] <= 4]
    hari_pupuk   = [d for d in daily if d["prediksi_cuaca"] == "sun" and d["precipitation"] < 1]
    hari_jamur   = [d for d in daily if d["prediksi_cuaca"] in ["rain", "drizzle"]]

    jml_tanam = len(hari_tanam)
    if jml_tanam == 0:
        hasil.append("❌ " + lg["tidak_ada_hari_cocok"])
    elif jml_tanam < 7:
        hasil.append("⚠️ " + lg["hari_cocok_sedikit"].format(jml_tanam))
    else:
        hasil.append("✅ " + lg["hari_cocok_banyak"].format(jml_tanam, total))

    jml_siram = len(hari_siram)
    if jml_siram < 5:
        hasil.append("❌ " + lg["hampir_tidak_perlu_siram"])
    elif jml_siram < 15:
        hasil.append("⚠️ " + lg["penyiraman_diperlukan"].format(jml_siram))
    else:
        hasil.append("✅ " + lg["penyiraman_rutin"].format(jml_siram))

    jml_semprot = len(hari_semprot)
    if jml_semprot < 10:
        hasil.append("⚠️ " + lg["hari_aman_semprot_sedikit"].format(jml_semprot))
    else:
        hasil.append("✅ " + lg["hari_aman_semprot"].format(jml_semprot))

    jml_pupuk = len(hari_pupuk)
    if jml_pupuk > 0:
        hasil.append("🌞 " + lg["hari_cerah_pupuk"].format(jml_pupuk))

    jml_jamur = len(hari_jamur)
    if jml_jamur > 15:
        hasil.append("⚠️ " + lg["potensi_jamur_tinggi"].format(jml_jamur))
    elif jml_jamur > 5:
        hasil.append("⚠️ " + lg["potensi_penyakit"].format(jml_jamur))

    return hasil


# ==== Peringatan ====
def peringatan(weather):
    lg = get_lang()
    hasil = []
    hujan = weather.get("precipitation", 0)
    angin = weather.get("wind", 0)
    cuaca = weather.get("prediksi_cuaca", "")

    if hujan > 30:
        hasil.append("🚨 " + lg["risiko_banjir"])
        hasil.append("⚠️ " + lg["potensi_penyakit_tinggi"])
    elif hujan > 10:
        hasil.append("⚠️ " + lg["potensi_jamur_meningkat"])
    if cuaca in ["rain", "drizzle"]:
        hasil.append("⚠️ " + lg["kelembaban_tinggi"])
    if angin > 7:
        hasil.append("🚨 " + lg["angin_kencang_roboh"])
    elif angin > 4:
        hasil.append("⚠️ " + lg["angin_cukup_kencang"])

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

    avg_precip   = round(sum(d["precipitation"] for d in daily) / len(daily), 2)
    avg_temp     = round(sum(d["temp_max"]      for d in daily) / len(daily), 2)
    avg_wind     = round(sum(d["wind"]          for d in daily) / len(daily), 2)
    total_precip = round(sum(d["precipitation"] for d in daily), 2)

    good_days = sum(
        1 for d in daily
        if d["prediksi_cuaca"] in ["sun", "fog"] and d["precipitation"] < 5
    )

    cuaca_count    = Counter(d["prediksi_cuaca"] for d in daily)
    dominant_cuaca = cuaca_count.most_common(1)[0][0]

    avg_weather = {
        "precipitation":  avg_precip,
        "temp_max":       avg_temp,
        "wind":           avg_wind,
        "prediksi_cuaca": dominant_cuaca
    }
    ranking  = hitung_saw(avg_weather)
    kegiatan = rekomendasi_kegiatan_30(daily)
    warning  = peringatan(avg_weather)

    for i, item in enumerate(ranking):
        item["bisa_panen"] = (
            item["skor"] >= 0.65
            and good_days >= 12
            and avg_precip <= 8.0
            and dominant_cuaca != "rain"
        )
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
    
    lg = get_lang()
    
    if request.method == 'POST':
        u = request.form['username']
        p = request.form['password']
        if u in users and users[u] == p:
            session['logged_in'] = True
            session['username']  = u
            session['lang'] = session.get('lang', 'id')
            return redirect(url_for('user_dashboard'))
        else:
            return render_template('login.html', error=lg['login_error'], lang=lg)
    
    return render_template('login.html', lang=lg)

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
        lg = get_lang()
        result = {}
        for nama, nilai in tanaman_data.items():
            nama_terjemahan = lg.get(f"tanaman_{nama.lower()}", nama)
            result[nama_terjemahan] = nilai
        return jsonify(result)
    
    if request.method == 'POST':
        data = request.get_json()
        nama = data.get('nama')
        if not nama:
            return jsonify({"error": "Nama tanaman diperlukan"}), 400
        
        nama_key = nama.lower().replace(" ", "_")
        if nama_key in tanaman_data:
            return jsonify({"error": "Tanaman sudah ada"}), 400
        
        tanaman_data[nama_key] = {
            "precipitation": float(data.get('precipitation', 0)),
            "temp_max": float(data.get('temp_max', 25)),
            "wind": float(data.get('wind', 2)),
            "cuaca_cocok": data.get('cuaca_cocok', ['sun'])
        }
        return jsonify({"message": f"Tanaman {nama} berhasil ditambahkan", "data": tanaman_data[nama_key]})
    
    if request.method == 'DELETE':
        data = request.get_json()
        nama = data.get('nama')
        if not nama:
            return jsonify({"error": "Nama tanaman diperlukan"}), 400
        
        nama_key = nama.lower().replace(" ", "_")
        if nama_key not in tanaman_data:
            return jsonify({"error": "Tanaman tidak ditemukan"}), 404
        
        del tanaman_data[nama_key]
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