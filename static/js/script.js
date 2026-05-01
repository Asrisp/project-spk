const weatherData = {
    'sun': {
        className: 'cerah',
        text: `- Sun - <span></span><i class="fas fa-sun"></i>`,
        details: `"Panasnya siang ini bikin aku sadar: hidup sudah cukup sulit tanpa harus meleleh."`
    },
    'rain': {
        className: 'hujan',
        text: `- Rain - <span></span><i class="fas fa-cloud-showers-heavy"></i>`,
        details: `"Hujan bikin ngantuk, tugas bikin panik. Kombinasi yang tidak manusiawi."`
    },
    'drizzle': {
        className: 'gerimis',
        text: `- Drizzle - <span></span><i class="fas fa-cloud-sun-rain"></i>`,
        details: `"Gerimis sambil ngopi itu enak, tapi ngopi sambil mikir tugas… beda cerita."`
    },
    'fog': {
        className: 'kabut',
        text: `- Fog - <span></span><i class="fas fa-smog"></i>`,
        details: `"Jalannya berkabut, seperti tugasku… sama-sama nggak kelihatan ujungnya."`
    },
    'snow': {
        className: 'salju',
        text: `- Snow - <span></span><i class="fas fa-snowflake"></i>`,
        details: `"Salju turun lembut… beda sama tugas yang turun bertubi-tubi."`
    }
};

function updateWeatherDisplay(weather) {
    const body = document.getElementById('weather-body');
    const weatherText = document.getElementById('weather-text');
    const weatherDetails = document.getElementById('weather-details');
    
    for (let key in weatherData) {
        body.classList.remove(weatherData[key].className);
    }
    
    const data = weatherData[weather.toLowerCase()];
    
    if (data) {
        body.classList.add(data.className);
        weatherText.innerHTML = data.text;
        weatherDetails.innerHTML = `<p>${data.details}</p>`;
    } else {
        body.classList.add('default');
        weatherText.innerHTML = `- Cuaca tidak diketahui -`;
        weatherDetails.innerHTML = `<p>Maaf, kami tidak memiliki data visual untuk cuaca ini.</p>`;
    }
}

function initializeWeather(weather) {
    updateWeatherDisplay(weather);
}

async function predictWeather() {
    const dateInput = document.getElementById('searchDate').value;
    const resultDiv = document.getElementById('prediction-result');
    
    if (!dateInput) {
        resultDiv.innerHTML = "<p style='color: red;'>Harap pilih tanggal terlebih dahulu.</p>";
        return;
    }

    try {
        const response = await fetch('/spk', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ date: dateInput }),
        });

        
        // FIX 1: Hapus deklarasi duplikat - hanya deklarasikan sekali
        const data = await response.json();

        console.log(data);

        // FIX 2: Cek error DULU sebelum akses data lainnya
        if (data.error) {
            resultDiv.innerHTML = `<p style='color: red;'>${data.error}</p>`;
            return;
        }

        const ranking = data.ranking_tanaman;
        const kegiatan = data.kegiatan;
        const peringatan = data.peringatan;

        if (!ranking || !kegiatan || !peringatan) {
            document.getElementById('spk-result').innerHTML =
                "<p style='color:red;'>Data SPK tidak tersedia</p>";
            return;
        }

        const tanggal = data.tanggal;
        const prediksi = data.prediksi_cuaca;

        resultDiv.innerHTML = `
            <div style="margin-top: 20px; padding: 15px; font-size: 12px; background: rgba(255,255,255,0.2); border-radius: 10px; backdrop-filter: blur(10px);">
                <div style="text-align: center;">
                    <p><strong>🌡️ Suhu Maksimum:</strong> ${data.temp_max}°C</p>
                    <p><strong>🌡️ Suhu Minimum:</strong> ${data.temp_min}°C</p>
                    <p><strong>💧 Curah Hujan:</strong> ${data.precipitation} inch</p>
                    <p><strong>💨 Kecepatan Angin:</strong> ${data.wind} mph</p>
                </div>
            </div>
        `;

        // === TAMPILKAN SPK ===
        let tanamanHTML = "<h4>🌱 Rekomendasi Tanaman:</h4><ul>";
        ranking.forEach(item => {
            tanamanHTML += `<li>${item.tanaman} (Skor: ${item.skor})</li>`;
        });
        tanamanHTML += "</ul>";

        let kegiatanHTML = "<h4>🚜 Kegiatan:</h4><ul>";
        kegiatan.forEach(item => {
            kegiatanHTML += `<li>${item}</li>`;
        });
        kegiatanHTML += "</ul>";

        let warningHTML = "<h4>⚠️ Peringatan:</h4><ul>";
        peringatan.forEach(item => {
            warningHTML += `<li>${item}</li>`;
        });
        warningHTML += "</ul>";

        document.getElementById('spk-result').innerHTML = `
            ${tanamanHTML}
            ${kegiatanHTML}
            ${warningHTML}
        `;

        updateWeatherDisplay(prediksi);

    } catch (error) {
        resultDiv.innerHTML = `<p style='color: red;'>Terjadi kesalahan saat melakukan prediksi.</p>`;
        console.error('Error:', error);
    }
}