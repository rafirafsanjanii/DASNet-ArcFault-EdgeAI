/*
 * DASNet - Distributed Arc-Signature Network
 * main.cpp
 *
 * Tim       : Xtal
 * Platform  : ESP32-S3 (ESP-IDF + FreeRTOS)
 * Target    : Deteksi dini arc fault DC PLTS atap
 *
 * Arsitektur firmware:
 *   1) Akuisisi sensor target: ACS712 (arus), DS18B20 (suhu), BH1750 (lux/iradiansi)
 *   2) Fusi proxy indicator:
 *        I_ripple = sigma(I_window) / mean(I_window)
 *        T_rate   = dT / dt
 *        eta_dev  = 1 - P_out_actual / (k * G * A)
 *        S        = w1*I_ripple + w2*T_rate + w3*eta_dev
 *      Status: NORMAL  S < 0.3
 *              WASPADA 0.3 <= S < 0.7
 *              BAHAYA  S >= 0.7
 *   3) Proteksi lokal: LED RGB + buzzer + relay (failsafe on-device)
 *   4) Telemetri LoRa SX1278 433 MHz ke gateway komunitas
 *   5) TensorFlow Lite Micro: load + benchmark/inference model INT8 ResNet 1D-CNN
 *
 * CATATAN ILMIAH PENTING
 * -----------------------
 * Model ResNet 1D-CNN pada naskah dilatih pada 3 kanal dataset sekunder:
 * current, voltage, acoustic_em dengan input (3, 1024). Sementara arsitektur
 * sensor fisik target memakai current, temperature, irradiance.
 *
 * Karena domain kanal tidak identik, firmware ini TIDAK secara default memakai
 * output TinyML sebagai pemicu relay pada data sensor fisik. Keputusan proteksi
 * live memakai skema fusi proxy rule-based. Model INT8 tetap dimuat dan dapat
 * dieksekusi on-device untuk validasi deployment/benchmark atau diberi window
 * 3-kanal yang kompatibel melalui fungsi tinyml_infer().
 *
 * Sebelum deployment fisik:
 *   - Kalibrasikan ACS712, konversi lux->W/m2, k efisiensi, luas panel, Vdc.
 *   - Ganti bobot w1/w2/w3 dengan hasil grid-search notebook.
 *   - Validasi pada testbed/HIL.
 *   - Jangan gunakan relay auto-trip sebagai pengganti AFCI tersertifikasi.
 *
 * Dependensi:
 *   - ESP-IDF 5.x
 *   - esp-tflite-micro / TensorFlow Lite Micro component
 *   - dasnet_model_data.h (hasil konversi model INT8)
 *
 * Jika simbol array pada dasnet_model_data.h berbeda, ubah dua macro:
 *   DASNET_MODEL_DATA
 *   DASNET_MODEL_DATA_LEN
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <new>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "driver/gpio.h"
#include "driver/i2c.h"
#include "driver/spi_master.h"

#include "esp_adc/adc_oneshot.h"
#include "esp_adc/adc_cali.h"
#include "esp_adc/adc_cali_scheme.h"

#include "esp_check.h"
#include "esp_err.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_rom_sys.h"
#include "esp_timer.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#include "dasnet_model_data.h"

// ============================================================================
// 1. KONFIGURASI UTAMA
// ============================================================================

static const char *TAG = "DASNet";

// ------------------------- Fitur ---------------------------------------------
#define DASNET_ENABLE_TINYML                1
#define DASNET_ENABLE_LORA                  1

// Demi keselamatan, default relay auto-trip OFF.
// Ubah menjadi 1 hanya setelah kalibrasi + validasi testbed/HIL.
#define DASNET_ENABLE_RELAY_TRIP            0

// TinyML tidak boleh menjadi dasar trip live sampai input live cocok
// dengan domain training (current, voltage, acoustic_em).
#define DASNET_ALLOW_TINYML_LIVE_TRIP       0

// ------------------------- Identitas node -----------------------------------
#define DASNET_NODE_ID                      1
#define DASNET_FW_VERSION                   "1.0.0"

// ------------------------- GPIO ESP32-S3 ------------------------------------
// Sesuaikan dengan PCB / DevKit yang dipakai.
#define PIN_ACS712_ADC                       GPIO_NUM_1
#define ADC_ACS712_CHANNEL                   ADC_CHANNEL_0
#define ADC_ACS712_UNIT                      ADC_UNIT_1

#define PIN_DS18B20                          GPIO_NUM_2

#define PIN_I2C_SDA                          GPIO_NUM_8
#define PIN_I2C_SCL                          GPIO_NUM_9
#define I2C_PORT                             I2C_NUM_0
#define I2C_FREQ_HZ                          100000

#define PIN_RELAY                            GPIO_NUM_10
#define PIN_LED_R                            GPIO_NUM_11
#define PIN_LED_G                            GPIO_NUM_12
#define PIN_LED_B                            GPIO_NUM_13
#define PIN_BUZZER                           GPIO_NUM_14

#define PIN_LORA_SCK                         GPIO_NUM_5
#define PIN_LORA_MISO                        GPIO_NUM_6
#define PIN_LORA_MOSI                        GPIO_NUM_7
#define PIN_LORA_CS                          GPIO_NUM_15
#define PIN_LORA_RST                         GPIO_NUM_16
#define PIN_LORA_DIO0                        GPIO_NUM_17
#define LORA_SPI_HOST                        SPI2_HOST

// ------------------------- Level logika output ------------------------------
#define RELAY_TRIP_LEVEL                     1
#define RELAY_SAFE_LEVEL                     0
#define LED_ON_LEVEL                         1
#define LED_OFF_LEVEL                        0
#define BUZZER_ON_LEVEL                      1
#define BUZZER_OFF_LEVEL                     0

// ------------------------- Akuisisi proxy -----------------------------------
#define CURRENT_SAMPLE_HZ                    200
#define CURRENT_SAMPLE_PERIOD_MS             (1000 / CURRENT_SAMPLE_HZ)
#define PROXY_WINDOW_MS                      1000

// Filter EMA untuk suhu/lux (bukan untuk ripple arus mentah).
#define EMA_ALPHA                            0.25f

// ------------------------- Kalibrasi sensor ---------------------------------
// WAJIB disesuaikan dengan rangkaian nyata.
//
// ACS712:
// Nilai berikut diasumsikan adalah tegangan DI SISI ADC ESP32 setelah
// conditioning/divider, bukan langsung output 5V ACS712.
// Jangan pernah memberi ADC ESP32 tegangan di atas batas I/O yang diizinkan.
#define ACS712_ZERO_MV                       1650.0f
#define ACS712_SENSITIVITY_MV_PER_A          66.0f

// BH1750 mengukur lux. Konversi lux -> W/m2 bersifat spektral dan perlu
// dikalibrasi terhadap irradiance reference/pyranometer.
// Nilai ini hanya initial engineering approximation.
#define LUX_TO_WM2_FACTOR                    0.0079f

// Parameter PLTS untuk eta_dev. Kalibrasikan terhadap panel aktual.
#define PV_NOMINAL_VOLTAGE_V                 36.0f
#define PANEL_EFFICIENCY_K                   0.20f
#define PANEL_EFFECTIVE_AREA_M2              2.00f

// ------------------------- Bobot fusi proxy ---------------------------------
// Naskah menyatakan bobot diperoleh via grid search, tetapi nilai numeriknya
// tidak dicantumkan di PDF. Default berikut hanya placeholder ter-normalisasi.
// GANTI dengan bobot final dari notebook/hasil grid-search tim Xtal.
#define PROXY_WEIGHT_RIPPLE                  0.34f
#define PROXY_WEIGHT_TEMP_RATE               0.33f
#define PROXY_WEIGHT_ETA_DEV                 0.33f

// Ambang status dari naskah.
#define SCORE_THRESHOLD_WARNING              0.30f
#define SCORE_THRESHOLD_DANGER               0.70f

// ------------------------- LoRa SX1278 --------------------------------------
#define LORA_FREQUENCY_HZ                    433000000UL
#define LORA_TX_POWER_DBM                    17
#define LORA_TX_TIMEOUT_MS                   1500

// ------------------------- TinyML -------------------------------------------
#define TENSOR_ARENA_BYTES                   (2 * 1024 * 1024)

// Jika header hasil generator memakai nama simbol berbeda, ubah dua macro ini.
// Contoh xxd -i mungkin menghasilkan: dasnet_int8_tflite / dasnet_int8_tflite_len.
#ifndef DASNET_MODEL_DATA
#define DASNET_MODEL_DATA                    dasnet_model_data
#endif

#ifndef DASNET_MODEL_DATA_LEN
#define DASNET_MODEL_DATA_LEN                dasnet_model_data_len
#endif

// ============================================================================
// 2. TIPE DATA DAN STATE GLOBAL
// ============================================================================

enum class DasnetStatus : uint8_t {
    NORMAL  = 0,
    WASPADA = 1,
    BAHAYA  = 2,
    ERROR   = 3
};

struct SensorSnapshot {
    float current_a;
    float temperature_c;
    float illuminance_lux;
    float irradiance_wm2;
    bool current_ok;
    bool temperature_ok;
    bool irradiance_ok;
};

struct ProxyResult {
    float current_mean_a;
    float current_std_a;
    float ripple;
    float temp_rate_c_per_s;
    float eta_dev;
    float score;
    DasnetStatus status;
};

struct CurrentAccumulator {
    double sum;
    double sum_sq;
    uint32_t count;
    float latest;
};

static portMUX_TYPE s_current_mux = portMUX_INITIALIZER_UNLOCKED;
static CurrentAccumulator s_current_acc{};

static SensorSnapshot s_sensor{};
static ProxyResult s_proxy{};

static adc_oneshot_unit_handle_t s_adc_handle = nullptr;
static adc_cali_handle_t s_adc_cali = nullptr;
static bool s_adc_calibrated = false;

static spi_device_handle_t s_lora_spi = nullptr;
static uint32_t s_packet_seq = 0;

// TinyML
#if DASNET_ENABLE_TINYML
static tflite::MicroInterpreter *s_interpreter = nullptr;
static TfLiteTensor *s_input = nullptr;
static TfLiteTensor *s_output = nullptr;
static uint8_t *s_tensor_arena = nullptr;
static bool s_tinyml_ready = false;
#endif

// ============================================================================
// 3. UTILITAS
// ============================================================================

static inline float clampf(float x, float lo, float hi) {
    return std::max(lo, std::min(x, hi));
}

static const char *status_to_string(DasnetStatus s) {
    switch (s) {
        case DasnetStatus::NORMAL:  return "NORMAL";
        case DasnetStatus::WASPADA: return "WASPADA";
        case DasnetStatus::BAHAYA:  return "BAHAYA";
        default:                    return "ERROR";
    }
}

static uint16_t crc16_ccitt(const uint8_t *data, size_t len) {
    uint16_t crc = 0xFFFF;
    for (size_t i = 0; i < len; ++i) {
        crc ^= static_cast<uint16_t>(data[i]) << 8;
        for (int b = 0; b < 8; ++b) {
            crc = (crc & 0x8000) ? static_cast<uint16_t>((crc << 1) ^ 0x1021)
                                 : static_cast<uint16_t>(crc << 1);
        }
    }
    return crc;
}

// ============================================================================
// 4. GPIO / OUTPUT PROTEKSI
// ============================================================================

static esp_err_t init_outputs() {
    gpio_config_t cfg{};
    cfg.mode = GPIO_MODE_OUTPUT;
    cfg.pin_bit_mask =
        (1ULL << PIN_RELAY) |
        (1ULL << PIN_LED_R) |
        (1ULL << PIN_LED_G) |
        (1ULL << PIN_LED_B) |
        (1ULL << PIN_BUZZER);
    ESP_RETURN_ON_ERROR(gpio_config(&cfg), TAG, "gpio_config output gagal");

    gpio_set_level(PIN_RELAY, RELAY_SAFE_LEVEL);
    gpio_set_level(PIN_LED_R, LED_OFF_LEVEL);
    gpio_set_level(PIN_LED_G, LED_OFF_LEVEL);
    gpio_set_level(PIN_LED_B, LED_OFF_LEVEL);
    gpio_set_level(PIN_BUZZER, BUZZER_OFF_LEVEL);
    return ESP_OK;
}

static void set_rgb(bool r, bool g, bool b) {
    gpio_set_level(PIN_LED_R, r ? LED_ON_LEVEL : LED_OFF_LEVEL);
    gpio_set_level(PIN_LED_G, g ? LED_ON_LEVEL : LED_OFF_LEVEL);
    gpio_set_level(PIN_LED_B, b ? LED_ON_LEVEL : LED_OFF_LEVEL);
}

static void apply_local_protection(DasnetStatus status) {
    switch (status) {
        case DasnetStatus::NORMAL:
            set_rgb(false, true, false);
            gpio_set_level(PIN_BUZZER, BUZZER_OFF_LEVEL);
            gpio_set_level(PIN_RELAY, RELAY_SAFE_LEVEL);
            break;

        case DasnetStatus::WASPADA:
            set_rgb(true, true, false);
            // Task indikator akan membuat pola beep; default di sini OFF.
            gpio_set_level(PIN_BUZZER, BUZZER_OFF_LEVEL);
            gpio_set_level(PIN_RELAY, RELAY_SAFE_LEVEL);
            break;

        case DasnetStatus::BAHAYA:
            set_rgb(true, false, false);
            gpio_set_level(PIN_BUZZER, BUZZER_ON_LEVEL);
#if DASNET_ENABLE_RELAY_TRIP
            gpio_set_level(PIN_RELAY, RELAY_TRIP_LEVEL);
#else
            gpio_set_level(PIN_RELAY, RELAY_SAFE_LEVEL);
#endif
            break;

        default:
            // Error sensor/firmware: tampilkan magenta, jangan trip otomatis.
            set_rgb(true, false, true);
            gpio_set_level(PIN_BUZZER, BUZZER_OFF_LEVEL);
            gpio_set_level(PIN_RELAY, RELAY_SAFE_LEVEL);
            break;
    }
}

// ============================================================================
// 5. ADC ACS712
// ============================================================================

static esp_err_t init_acs712_adc() {
    adc_oneshot_unit_init_cfg_t unit_cfg{};
    unit_cfg.unit_id = ADC_ACS712_UNIT;
    ESP_RETURN_ON_ERROR(adc_oneshot_new_unit(&unit_cfg, &s_adc_handle),
                        TAG, "adc_oneshot_new_unit gagal");

    adc_oneshot_chan_cfg_t chan_cfg{};
    chan_cfg.atten = ADC_ATTEN_DB_11;
    chan_cfg.bitwidth = ADC_BITWIDTH_DEFAULT;
    ESP_RETURN_ON_ERROR(adc_oneshot_config_channel(
                            s_adc_handle, ADC_ACS712_CHANNEL, &chan_cfg),
                        TAG, "adc channel config gagal");

#if ADC_CALI_SCHEME_CURVE_FITTING_SUPPORTED
    adc_cali_curve_fitting_config_t cali_cfg{};
    cali_cfg.unit_id = ADC_ACS712_UNIT;
    cali_cfg.chan = ADC_ACS712_CHANNEL;
    cali_cfg.atten = ADC_ATTEN_DB_11;
    cali_cfg.bitwidth = ADC_BITWIDTH_DEFAULT;

    if (adc_cali_create_scheme_curve_fitting(&cali_cfg, &s_adc_cali) == ESP_OK) {
        s_adc_calibrated = true;
        ESP_LOGI(TAG, "ADC calibration: curve fitting aktif");
    } else {
        ESP_LOGW(TAG, "ADC calibration tidak tersedia; fallback aproksimasi raw->mV");
    }
#endif
    return ESP_OK;
}

static bool read_current_a(float *out_current_a) {
    if (!out_current_a || !s_adc_handle) return false;

    int raw = 0;
    if (adc_oneshot_read(s_adc_handle, ADC_ACS712_CHANNEL, &raw) != ESP_OK) {
        return false;
    }

    int mv = 0;
    if (s_adc_calibrated) {
        if (adc_cali_raw_to_voltage(s_adc_cali, raw, &mv) != ESP_OK) {
            return false;
        }
    } else {
        // Aproksimasi 12-bit, full-scale ~3.3V; hanya fallback.
        mv = static_cast<int>((raw / 4095.0f) * 3300.0f);
    }

    float current = (static_cast<float>(mv) - ACS712_ZERO_MV)
                    / ACS712_SENSITIVITY_MV_PER_A;

    // Jalur PV DC pada use-case ini diasumsikan arus maju.
    if (current < 0.0f) current = 0.0f;
    *out_current_a = current;
    return std::isfinite(current);
}

// ============================================================================
// 6. DS18B20 - IMPLEMENTASI 1-WIRE MINIMAL TANPA DEPENDENSI EKSTERNAL
// ============================================================================

static inline void ow_drive_low() {
    gpio_set_direction(PIN_DS18B20, GPIO_MODE_OUTPUT_OD);
    gpio_set_level(PIN_DS18B20, 0);
}

static inline void ow_release() {
    gpio_set_direction(PIN_DS18B20, GPIO_MODE_INPUT);
}

static bool ow_reset() {
    ow_drive_low();
    esp_rom_delay_us(480);
    ow_release();
    esp_rom_delay_us(70);
    int presence = gpio_get_level(PIN_DS18B20);
    esp_rom_delay_us(410);
    return presence == 0;
}

static void ow_write_bit(uint8_t bit) {
    ow_drive_low();
    if (bit) {
        esp_rom_delay_us(6);
        ow_release();
        esp_rom_delay_us(64);
    } else {
        esp_rom_delay_us(60);
        ow_release();
        esp_rom_delay_us(10);
    }
}

static uint8_t ow_read_bit() {
    ow_drive_low();
    esp_rom_delay_us(6);
    ow_release();
    esp_rom_delay_us(9);
    uint8_t bit = static_cast<uint8_t>(gpio_get_level(PIN_DS18B20));
    esp_rom_delay_us(55);
    return bit;
}

static void ow_write_byte(uint8_t v) {
    for (int i = 0; i < 8; ++i) {
        ow_write_bit(v & 0x01);
        v >>= 1;
    }
}

static uint8_t ow_read_byte() {
    uint8_t v = 0;
    for (int i = 0; i < 8; ++i) {
        v |= static_cast<uint8_t>(ow_read_bit() << i);
    }
    return v;
}

static uint8_t ds18b20_crc8(const uint8_t *data, size_t len) {
    uint8_t crc = 0;
    for (size_t i = 0; i < len; ++i) {
        uint8_t in = data[i];
        for (int j = 0; j < 8; ++j) {
            uint8_t mix = (crc ^ in) & 0x01;
            crc >>= 1;
            if (mix) crc ^= 0x8C;
            in >>= 1;
        }
    }
    return crc;
}

static esp_err_t init_ds18b20() {
    gpio_config_t cfg{};
    cfg.mode = GPIO_MODE_INPUT;
    cfg.pull_up_en = GPIO_PULLUP_ENABLE;
    cfg.pin_bit_mask = 1ULL << PIN_DS18B20;
    ESP_RETURN_ON_ERROR(gpio_config(&cfg), TAG, "DS18B20 gpio config gagal");

    if (!ow_reset()) {
        ESP_LOGW(TAG, "DS18B20 tidak terdeteksi saat init");
        return ESP_ERR_NOT_FOUND;
    }
    return ESP_OK;
}

static bool read_temperature_c(float *out_temp_c) {
    if (!out_temp_c) return false;

    // Skip ROM (single sensor), Convert T
    if (!ow_reset()) return false;
    ow_write_byte(0xCC);
    ow_write_byte(0x44);

    // Resolusi default 12-bit: maksimum ~750 ms.
    vTaskDelay(pdMS_TO_TICKS(750));

    if (!ow_reset()) return false;
    ow_write_byte(0xCC);
    ow_write_byte(0xBE);

    uint8_t scratch[9]{};
    for (auto &b : scratch) b = ow_read_byte();

    if (ds18b20_crc8(scratch, 8) != scratch[8]) {
        return false;
    }

    int16_t raw = static_cast<int16_t>(
        (static_cast<uint16_t>(scratch[1]) << 8) | scratch[0]);

    float temp = raw / 16.0f;
    if (!std::isfinite(temp) || temp < -55.0f || temp > 125.0f) return false;

    *out_temp_c = temp;
    return true;
}

// ============================================================================
// 7. BH1750 (I2C)
// ============================================================================

static constexpr uint8_t BH1750_ADDR = 0x23;
static constexpr uint8_t BH1750_POWER_ON = 0x01;
static constexpr uint8_t BH1750_CONT_H_RES_MODE = 0x10;

static esp_err_t init_i2c_and_bh1750() {
    i2c_config_t cfg{};
    cfg.mode = I2C_MODE_MASTER;
    cfg.sda_io_num = PIN_I2C_SDA;
    cfg.scl_io_num = PIN_I2C_SCL;
    cfg.sda_pullup_en = GPIO_PULLUP_ENABLE;
    cfg.scl_pullup_en = GPIO_PULLUP_ENABLE;
    cfg.master.clk_speed = I2C_FREQ_HZ;

    ESP_RETURN_ON_ERROR(i2c_param_config(I2C_PORT, &cfg), TAG, "i2c_param_config gagal");
    ESP_RETURN_ON_ERROR(
        i2c_driver_install(I2C_PORT, cfg.mode, 0, 0, 0),
        TAG, "i2c_driver_install gagal");

    uint8_t cmd = BH1750_POWER_ON;
    ESP_RETURN_ON_ERROR(
        i2c_master_write_to_device(I2C_PORT, BH1750_ADDR, &cmd, 1,
                                   pdMS_TO_TICKS(100)),
        TAG, "BH1750 power-on gagal");

    cmd = BH1750_CONT_H_RES_MODE;
    ESP_RETURN_ON_ERROR(
        i2c_master_write_to_device(I2C_PORT, BH1750_ADDR, &cmd, 1,
                                   pdMS_TO_TICKS(100)),
        TAG, "BH1750 mode set gagal");

    vTaskDelay(pdMS_TO_TICKS(180));
    return ESP_OK;
}

static bool read_bh1750_lux(float *out_lux) {
    if (!out_lux) return false;

    uint8_t data[2]{};
    esp_err_t err = i2c_master_read_from_device(
        I2C_PORT, BH1750_ADDR, data, sizeof(data), pdMS_TO_TICKS(100));

    if (err != ESP_OK) return false;

    uint16_t raw = static_cast<uint16_t>((data[0] << 8) | data[1]);
    float lux = raw / 1.2f;
    if (!std::isfinite(lux) || lux < 0.0f) return false;

    *out_lux = lux;
    return true;
}

// ============================================================================
// 8. LORA SX1278 - DRIVER TX MINIMAL
// ============================================================================

#if DASNET_ENABLE_LORA

namespace sx1278 {
static constexpr uint8_t REG_FIFO                 = 0x00;
static constexpr uint8_t REG_OP_MODE              = 0x01;
static constexpr uint8_t REG_FRF_MSB              = 0x06;
static constexpr uint8_t REG_FRF_MID              = 0x07;
static constexpr uint8_t REG_FRF_LSB              = 0x08;
static constexpr uint8_t REG_PA_CONFIG            = 0x09;
static constexpr uint8_t REG_LNA                  = 0x0C;
static constexpr uint8_t REG_FIFO_ADDR_PTR        = 0x0D;
static constexpr uint8_t REG_FIFO_TX_BASE_ADDR    = 0x0E;
static constexpr uint8_t REG_IRQ_FLAGS            = 0x12;
static constexpr uint8_t REG_MODEM_CONFIG_1       = 0x1D;
static constexpr uint8_t REG_MODEM_CONFIG_2       = 0x1E;
static constexpr uint8_t REG_PREAMBLE_MSB         = 0x20;
static constexpr uint8_t REG_PREAMBLE_LSB         = 0x21;
static constexpr uint8_t REG_PAYLOAD_LENGTH       = 0x22;
static constexpr uint8_t REG_MODEM_CONFIG_3       = 0x26;
static constexpr uint8_t REG_DIO_MAPPING_1        = 0x40;
static constexpr uint8_t REG_VERSION              = 0x42;

static constexpr uint8_t MODE_LONG_RANGE          = 0x80;
static constexpr uint8_t MODE_SLEEP               = 0x00;
static constexpr uint8_t MODE_STDBY               = 0x01;
static constexpr uint8_t MODE_TX                  = 0x03;

static constexpr uint8_t IRQ_TX_DONE_MASK         = 0x08;
}

static esp_err_t lora_write_reg(uint8_t reg, uint8_t value) {
    uint8_t tx[2] = { static_cast<uint8_t>(reg | 0x80), value };
    spi_transaction_t t{};
    t.length = 16;
    t.tx_buffer = tx;
    return spi_device_transmit(s_lora_spi, &t);
}

static esp_err_t lora_read_reg(uint8_t reg, uint8_t *value) {
    if (!value) return ESP_ERR_INVALID_ARG;
    uint8_t tx[2] = { static_cast<uint8_t>(reg & 0x7F), 0x00 };
    uint8_t rx[2]{};
    spi_transaction_t t{};
    t.length = 16;
    t.tx_buffer = tx;
    t.rx_buffer = rx;
    esp_err_t err = spi_device_transmit(s_lora_spi, &t);
    if (err == ESP_OK) *value = rx[1];
    return err;
}

static esp_err_t lora_write_fifo(const uint8_t *data, size_t len) {
    if (!data || len == 0 || len > 255) return ESP_ERR_INVALID_ARG;

    uint8_t *buf = static_cast<uint8_t *>(heap_caps_malloc(len + 1, MALLOC_CAP_8BIT));
    if (!buf) return ESP_ERR_NO_MEM;

    buf[0] = sx1278::REG_FIFO | 0x80;
    memcpy(buf + 1, data, len);

    spi_transaction_t t{};
    t.length = (len + 1) * 8;
    t.tx_buffer = buf;
    esp_err_t err = spi_device_transmit(s_lora_spi, &t);

    heap_caps_free(buf);
    return err;
}

static esp_err_t init_lora() {
    spi_bus_config_t bus_cfg{};
    bus_cfg.sclk_io_num = PIN_LORA_SCK;
    bus_cfg.mosi_io_num = PIN_LORA_MOSI;
    bus_cfg.miso_io_num = PIN_LORA_MISO;
    bus_cfg.quadwp_io_num = -1;
    bus_cfg.quadhd_io_num = -1;
    bus_cfg.max_transfer_sz = 260;

    ESP_RETURN_ON_ERROR(spi_bus_initialize(LORA_SPI_HOST, &bus_cfg, SPI_DMA_CH_AUTO),
                        TAG, "spi_bus_initialize LoRa gagal");

    spi_device_interface_config_t dev_cfg{};
    dev_cfg.clock_speed_hz = 8 * 1000 * 1000;
    dev_cfg.mode = 0;
    dev_cfg.spics_io_num = PIN_LORA_CS;
    dev_cfg.queue_size = 1;

    ESP_RETURN_ON_ERROR(spi_bus_add_device(LORA_SPI_HOST, &dev_cfg, &s_lora_spi),
                        TAG, "spi_bus_add_device LoRa gagal");

    gpio_config_t out_cfg{};
    out_cfg.mode = GPIO_MODE_OUTPUT;
    out_cfg.pin_bit_mask = 1ULL << PIN_LORA_RST;
    ESP_RETURN_ON_ERROR(gpio_config(&out_cfg), TAG, "gpio LoRa RST gagal");

    gpio_config_t in_cfg{};
    in_cfg.mode = GPIO_MODE_INPUT;
    in_cfg.pin_bit_mask = 1ULL << PIN_LORA_DIO0;
    ESP_RETURN_ON_ERROR(gpio_config(&in_cfg), TAG, "gpio LoRa DIO0 gagal");

    // Hardware reset
    gpio_set_level(PIN_LORA_RST, 0);
    vTaskDelay(pdMS_TO_TICKS(10));
    gpio_set_level(PIN_LORA_RST, 1);
    vTaskDelay(pdMS_TO_TICKS(10));

    uint8_t version = 0;
    ESP_RETURN_ON_ERROR(lora_read_reg(sx1278::REG_VERSION, &version),
                        TAG, "read SX1278 version gagal");
    if (version != 0x12) {
        ESP_LOGE(TAG, "SX1278 version invalid: 0x%02X (expected 0x12)", version);
        return ESP_ERR_NOT_FOUND;
    }

    // LoRa mode + sleep.
    ESP_RETURN_ON_ERROR(
        lora_write_reg(sx1278::REG_OP_MODE,
                       sx1278::MODE_LONG_RANGE | sx1278::MODE_SLEEP),
        TAG, "LoRa sleep gagal");

    // Frequency register: FRF = Freq / (32MHz / 2^19)
    uint64_t frf = (static_cast<uint64_t>(LORA_FREQUENCY_HZ) << 19) / 32000000ULL;
    lora_write_reg(sx1278::REG_FRF_MSB, static_cast<uint8_t>(frf >> 16));
    lora_write_reg(sx1278::REG_FRF_MID, static_cast<uint8_t>(frf >> 8));
    lora_write_reg(sx1278::REG_FRF_LSB, static_cast<uint8_t>(frf));

    // FIFO base
    lora_write_reg(sx1278::REG_FIFO_TX_BASE_ADDR, 0x00);

    // LNA boost
    uint8_t lna = 0;
    lora_read_reg(sx1278::REG_LNA, &lna);
    lora_write_reg(sx1278::REG_LNA, lna | 0x03);

    // BW 125kHz, CR 4/5, explicit header.
    lora_write_reg(sx1278::REG_MODEM_CONFIG_1, 0x72);
    // SF7 + CRC ON.
    lora_write_reg(sx1278::REG_MODEM_CONFIG_2, 0x74);
    // AGC auto.
    lora_write_reg(sx1278::REG_MODEM_CONFIG_3, 0x04);

    // Preamble 8 symbols.
    lora_write_reg(sx1278::REG_PREAMBLE_MSB, 0x00);
    lora_write_reg(sx1278::REG_PREAMBLE_LSB, 0x08);

    // PA_BOOST, approx 2..17 dBm.
    int pwr = std::max(2, std::min(LORA_TX_POWER_DBM, 17));
    lora_write_reg(sx1278::REG_PA_CONFIG,
                   static_cast<uint8_t>(0x80 | (pwr - 2)));

    // DIO0 = TxDone in TX mode.
    lora_write_reg(sx1278::REG_DIO_MAPPING_1, 0x40);

    // Standby.
    lora_write_reg(sx1278::REG_OP_MODE,
                   sx1278::MODE_LONG_RANGE | sx1278::MODE_STDBY);

    ESP_LOGI(TAG, "LoRa SX1278 siap @ %.1f MHz", LORA_FREQUENCY_HZ / 1e6);
    return ESP_OK;
}

static esp_err_t lora_send(const uint8_t *data, size_t len) {
    if (!s_lora_spi || !data || len == 0 || len > 255) return ESP_ERR_INVALID_ARG;

    // Standby + clear IRQ.
    lora_write_reg(sx1278::REG_OP_MODE,
                   sx1278::MODE_LONG_RANGE | sx1278::MODE_STDBY);
    lora_write_reg(sx1278::REG_IRQ_FLAGS, 0xFF);

    lora_write_reg(sx1278::REG_FIFO_ADDR_PTR, 0x00);
    ESP_RETURN_ON_ERROR(lora_write_fifo(data, len), TAG, "LoRa FIFO write gagal");
    lora_write_reg(sx1278::REG_PAYLOAD_LENGTH, static_cast<uint8_t>(len));

    // TX
    lora_write_reg(sx1278::REG_OP_MODE,
                   sx1278::MODE_LONG_RANGE | sx1278::MODE_TX);

    int64_t deadline = esp_timer_get_time() + (LORA_TX_TIMEOUT_MS * 1000LL);
    while (esp_timer_get_time() < deadline) {
        uint8_t flags = 0;
        if (lora_read_reg(sx1278::REG_IRQ_FLAGS, &flags) != ESP_OK) {
            return ESP_FAIL;
        }
        if (flags & sx1278::IRQ_TX_DONE_MASK) {
            lora_write_reg(sx1278::REG_IRQ_FLAGS, sx1278::IRQ_TX_DONE_MASK);
            lora_write_reg(sx1278::REG_OP_MODE,
                           sx1278::MODE_LONG_RANGE | sx1278::MODE_STDBY);
            return ESP_OK;
        }
        vTaskDelay(pdMS_TO_TICKS(5));
    }

    lora_write_reg(sx1278::REG_OP_MODE,
                   sx1278::MODE_LONG_RANGE | sx1278::MODE_STDBY);
    return ESP_ERR_TIMEOUT;
}
#endif

// ============================================================================
// 9. TENSORFLOW LITE MICRO - RESNET 1D-CNN INT8
// ============================================================================

#if DASNET_ENABLE_TINYML

static size_t tensor_element_count(const TfLiteTensor *t) {
    if (!t || !t->dims) return 0;
    size_t n = 1;
    for (int i = 0; i < t->dims->size; ++i) {
        n *= static_cast<size_t>(t->dims->data[i]);
    }
    return n;
}

static esp_err_t init_tinyml() {
    const tflite::Model *model = tflite::GetModel(DASNET_MODEL_DATA);
    if (!model) {
        ESP_LOGE(TAG, "TFLM GetModel gagal");
        return ESP_FAIL;
    }

    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "TFLite schema mismatch: model=%d runtime=%d",
                 model->version(), TFLITE_SCHEMA_VERSION);
        return ESP_ERR_INVALID_VERSION;
    }

    s_tensor_arena = static_cast<uint8_t *>(
        heap_caps_malloc(TENSOR_ARENA_BYTES, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));

    if (!s_tensor_arena) {
        ESP_LOGW(TAG, "PSRAM arena gagal, mencoba internal RAM");
        s_tensor_arena = static_cast<uint8_t *>(
            heap_caps_malloc(TENSOR_ARENA_BYTES, MALLOC_CAP_8BIT));
    }

    if (!s_tensor_arena) {
        ESP_LOGE(TAG, "Tidak cukup memori untuk tensor arena %u bytes",
                 static_cast<unsigned>(TENSOR_ARENA_BYTES));
        return ESP_ERR_NO_MEM;
    }

    static tflite::AllOpsResolver resolver;

    static tflite::MicroInterpreter interpreter(
        model, resolver, s_tensor_arena, TENSOR_ARENA_BYTES);

    s_interpreter = &interpreter;

    TfLiteStatus st = s_interpreter->AllocateTensors();
    if (st != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors gagal; naikkan TENSOR_ARENA_BYTES");
        return ESP_FAIL;
    }

    s_input = s_interpreter->input(0);
    s_output = s_interpreter->output(0);

    if (!s_input || !s_output) {
        ESP_LOGE(TAG, "Tensor input/output tidak ditemukan");
        return ESP_FAIL;
    }

    size_t in_count = tensor_element_count(s_input);
    size_t out_count = tensor_element_count(s_output);

    ESP_LOGI(TAG, "TinyML ready. Model bytes=%u, input elems=%u, output elems=%u",
             static_cast<unsigned>(DASNET_MODEL_DATA_LEN),
             static_cast<unsigned>(in_count),
             static_cast<unsigned>(out_count));

    // PDF mendeskripsikan input (3,1024) => 3072 elemen.
    if (in_count != 3U * 1024U) {
        ESP_LOGW(TAG,
                 "Input model bukan 3072 elemen. Firmware akan mengikuti tensor aktual.");
    }

    s_tinyml_ready = true;
    return ESP_OK;
}

static bool tensor_set_value(TfLiteTensor *t, size_t idx, float value) {
    if (!t || idx >= tensor_element_count(t)) return false;

    switch (t->type) {
        case kTfLiteFloat32:
            t->data.f[idx] = value;
            return true;

        case kTfLiteInt8: {
            const float scale = t->params.scale;
            const int zero = t->params.zero_point;
            if (scale <= 0.0f) return false;
            int q = static_cast<int>(std::lround(value / scale)) + zero;
            q = std::max(-128, std::min(127, q));
            t->data.int8[idx] = static_cast<int8_t>(q);
            return true;
        }

        case kTfLiteUInt8: {
            const float scale = t->params.scale;
            const int zero = t->params.zero_point;
            if (scale <= 0.0f) return false;
            int q = static_cast<int>(std::lround(value / scale)) + zero;
            q = std::max(0, std::min(255, q));
            t->data.uint8[idx] = static_cast<uint8_t>(q);
            return true;
        }

        default:
            return false;
    }
}

static float tensor_get_value(const TfLiteTensor *t, size_t idx) {
    if (!t || idx >= tensor_element_count(t)) return NAN;

    switch (t->type) {
        case kTfLiteFloat32:
            return t->data.f[idx];

        case kTfLiteInt8:
            return (static_cast<int>(t->data.int8[idx]) - t->params.zero_point)
                   * t->params.scale;

        case kTfLiteUInt8:
            return (static_cast<int>(t->data.uint8[idx]) - t->params.zero_point)
                   * t->params.scale;

        default:
            return NAN;
    }
}

/*
 * tinyml_infer()
 *
 * Input:
 *   flattened_input: tensor input flattened sesuai urutan model hasil notebook.
 *   count          : jumlah elemen; untuk arsitektur PDF seharusnya 3072.
 *
 * PENTING:
 *   Caller harus menerapkan preprocessing IDENTIK dengan training notebook,
 *   termasuk urutan kanal dan Z-score dari statistik TRAINING.
 *
 * Output class:
 *   0 = normal
 *   1 = series_arc
 *   2 = parallel_arc
 *
 * Fungsi ini tidak otomatis memicu relay.
 */
static esp_err_t tinyml_infer(const float *flattened_input,
                              size_t count,
                              int *out_class,
                              float out_scores[3],
                              int64_t *out_latency_us) {
    if (!s_tinyml_ready || !s_interpreter || !s_input || !s_output) {
        return ESP_ERR_INVALID_STATE;
    }
    if (!flattened_input || !out_class || !out_scores) {
        return ESP_ERR_INVALID_ARG;
    }

    size_t expected = tensor_element_count(s_input);
    if (count != expected) {
        ESP_LOGE(TAG, "tinyml_infer count=%u, expected=%u",
                 static_cast<unsigned>(count), static_cast<unsigned>(expected));
        return ESP_ERR_INVALID_SIZE;
    }

    for (size_t i = 0; i < count; ++i) {
        if (!tensor_set_value(s_input, i, flattened_input[i])) {
            return ESP_FAIL;
        }
    }

    int64_t t0 = esp_timer_get_time();
    TfLiteStatus st = s_interpreter->Invoke();
    int64_t t1 = esp_timer_get_time();

    if (out_latency_us) *out_latency_us = t1 - t0;

    if (st != kTfLiteOk) {
        ESP_LOGE(TAG, "TFLM Invoke gagal");
        return ESP_FAIL;
    }

    size_t out_n = tensor_element_count(s_output);
    if (out_n < 3) return ESP_ERR_INVALID_SIZE;

    int best = 0;
    float best_score = -INFINITY;

    for (int i = 0; i < 3; ++i) {
        out_scores[i] = tensor_get_value(s_output, i);
        if (out_scores[i] > best_score) {
            best_score = out_scores[i];
            best = i;
        }
    }

    *out_class = best;
    return ESP_OK;
}

static void tinyml_boot_benchmark() {
    if (!s_tinyml_ready || !s_input) return;

    const size_t n = tensor_element_count(s_input);
    float *dummy = static_cast<float *>(
        heap_caps_calloc(n, sizeof(float), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));

    if (!dummy) {
        dummy = static_cast<float *>(
            heap_caps_calloc(n, sizeof(float), MALLOC_CAP_8BIT));
    }
    if (!dummy) {
        ESP_LOGW(TAG, "Skip TinyML benchmark: no memory");
        return;
    }

    int pred = -1;
    float scores[3]{};
    int64_t latency_us = 0;
    esp_err_t err = tinyml_infer(dummy, n, &pred, scores, &latency_us);

    if (err == ESP_OK) {
        ESP_LOGI(TAG,
                 "TinyML boot benchmark (zero tensor): %.2f ms, pred=%d, scores=[%.4f %.4f %.4f]. "
                 "Prediksi dummy TIDAK bermakna secara ilmiah.",
                 latency_us / 1000.0, pred, scores[0], scores[1], scores[2]);
    }

    heap_caps_free(dummy);
}
#endif

// ============================================================================
// 10. FUSI PROXY INDICATOR
// ============================================================================

static DasnetStatus classify_proxy_score(float score) {
    if (!std::isfinite(score)) return DasnetStatus::ERROR;
    if (score >= SCORE_THRESHOLD_DANGER) return DasnetStatus::BAHAYA;
    if (score >= SCORE_THRESHOLD_WARNING) return DasnetStatus::WASPADA;
    return DasnetStatus::NORMAL;
}

static ProxyResult compute_proxy_result(double current_sum,
                                        double current_sum_sq,
                                        uint32_t current_count,
                                        float temp_c,
                                        float prev_temp_c,
                                        float dt_temp_s,
                                        float irradiance_wm2) {
    ProxyResult r{};
    r.status = DasnetStatus::ERROR;

    if (current_count == 0 || dt_temp_s <= 0.0f) {
        return r;
    }

    const double mean = current_sum / current_count;
    const double variance = std::max(
        0.0,
        (current_sum_sq / current_count) - (mean * mean));

    r.current_mean_a = static_cast<float>(mean);
    r.current_std_a = static_cast<float>(std::sqrt(variance));

    // I_ripple = sigma(I) / mean(I)
    const float denom = std::max(std::fabs(r.current_mean_a), 0.05f);
    r.ripple = r.current_std_a / denom;

    // T_rate = dT/dt; hanya kenaikan suhu yang dianggap indikasi anomali.
    r.temp_rate_c_per_s = std::max(0.0f, (temp_c - prev_temp_c) / dt_temp_s);

    // eta_dev = 1 - P_actual / (k * G * A)
    const float p_actual = std::max(0.0f, r.current_mean_a * PV_NOMINAL_VOLTAGE_V);
    const float p_expected = PANEL_EFFICIENCY_K
                           * std::max(irradiance_wm2, 0.0f)
                           * PANEL_EFFECTIVE_AREA_M2;

    if (p_expected > 1.0f) {
        r.eta_dev = 1.0f - (p_actual / p_expected);
        // Defisit efisiensi saja yang dianggap anomali; surplus/noise -> 0.
        r.eta_dev = clampf(r.eta_dev, 0.0f, 1.0f);
    } else {
        // Iradiansi terlalu rendah: indikator efisiensi tidak reliabel.
        r.eta_dev = 0.0f;
    }

    // Skor anomali komposit sesuai formulasi naskah.
    r.score = PROXY_WEIGHT_RIPPLE    * r.ripple
            + PROXY_WEIGHT_TEMP_RATE * r.temp_rate_c_per_s
            + PROXY_WEIGHT_ETA_DEV    * r.eta_dev;

    // Threshold naskah berada di domain [0,1].
    r.score = clampf(r.score, 0.0f, 1.0f);
    r.status = classify_proxy_score(r.score);

    return r;
}

// ============================================================================
// 11. PACKET TELEMETRI
// ============================================================================

static size_t build_status_packet(char *buf,
                                  size_t buf_size,
                                  const SensorSnapshot &s,
                                  const ProxyResult &p) {
    if (!buf || buf_size == 0) return 0;

    // JSON ringkas agar mudah dibaca gateway/dashboard.
    // CRC ditambahkan sebagai field terakhir, dihitung dari payload tanpa crc.
    char core[220]{};
    int core_len = snprintf(
        core, sizeof(core),
        "{\"node\":%u,\"seq\":%lu,\"ts_ms\":%lld,"
        "\"status\":\"%s\",\"score\":%.3f,"
        "\"i_mean\":%.3f,\"i_ripple\":%.4f,"
        "\"temp\":%.2f,\"temp_rate\":%.4f,"
        "\"lux\":%.1f,\"irr\":%.2f,\"eta_dev\":%.4f}",
        DASNET_NODE_ID,
        static_cast<unsigned long>(s_packet_seq),
        static_cast<long long>(esp_timer_get_time() / 1000),
        status_to_string(p.status),
        p.score,
        p.current_mean_a,
        p.ripple,
        s.temperature_c,
        p.temp_rate_c_per_s,
        s.illuminance_lux,
        s.irradiance_wm2,
        p.eta_dev);

    if (core_len <= 0 || static_cast<size_t>(core_len) >= sizeof(core)) return 0;

    uint16_t crc = crc16_ccitt(reinterpret_cast<const uint8_t *>(core),
                              static_cast<size_t>(core_len));

    // Ubah '}' terakhir menjadi field crc.
    core[core_len - 1] = '\0';
    int n = snprintf(buf, buf_size, "%s,\"crc16\":\"%04X\"}", core, crc);

    if (n <= 0 || static_cast<size_t>(n) >= buf_size) return 0;
    return static_cast<size_t>(n);
}

// ============================================================================
// 12. TASK FREERTOS
// ============================================================================

static void current_sampling_task(void *arg) {
    (void)arg;
    TickType_t last = xTaskGetTickCount();

    while (true) {
        float current_a = 0.0f;
        bool ok = read_current_a(&current_a);

        if (ok) {
            portENTER_CRITICAL(&s_current_mux);
            s_current_acc.sum += current_a;
            s_current_acc.sum_sq += static_cast<double>(current_a) * current_a;
            s_current_acc.count++;
            s_current_acc.latest = current_a;
            portEXIT_CRITICAL(&s_current_mux);

            s_sensor.current_a = current_a;
            s_sensor.current_ok = true;
        } else {
            s_sensor.current_ok = false;
        }

        vTaskDelayUntil(&last, pdMS_TO_TICKS(CURRENT_SAMPLE_PERIOD_MS));
    }
}

static void proxy_and_protection_task(void *arg) {
    (void)arg;

    bool first = true;
    float filtered_temp = NAN;
    float filtered_lux = NAN;
    float prev_temp = NAN;
    int64_t prev_temp_time_us = 0;

    while (true) {
        const int64_t cycle_start = esp_timer_get_time();

        // Ambil suhu.
        float temp = NAN;
        bool temp_ok = read_temperature_c(&temp);

        // Ambil lux.
        float lux = NAN;
        bool lux_ok = read_bh1750_lux(&lux);

        if (temp_ok) {
            filtered_temp = first || !std::isfinite(filtered_temp)
                          ? temp
                          : (EMA_ALPHA * temp + (1.0f - EMA_ALPHA) * filtered_temp);
            s_sensor.temperature_c = filtered_temp;
            s_sensor.temperature_ok = true;
        } else {
            s_sensor.temperature_ok = false;
        }

        if (lux_ok) {
            filtered_lux = first || !std::isfinite(filtered_lux)
                         ? lux
                         : (EMA_ALPHA * lux + (1.0f - EMA_ALPHA) * filtered_lux);
            s_sensor.illuminance_lux = filtered_lux;
            s_sensor.irradiance_wm2 = filtered_lux * LUX_TO_WM2_FACTOR;
            s_sensor.irradiance_ok = true;
        } else {
            s_sensor.irradiance_ok = false;
        }

        // Snapshot + reset accumulator arus untuk window berikutnya.
        CurrentAccumulator acc{};
        portENTER_CRITICAL(&s_current_mux);
        acc = s_current_acc;
        s_current_acc.sum = 0.0;
        s_current_acc.sum_sq = 0.0;
        s_current_acc.count = 0;
        portEXIT_CRITICAL(&s_current_mux);

        const int64_t now_us = esp_timer_get_time();
        float dt_s = (prev_temp_time_us > 0)
                   ? (now_us - prev_temp_time_us) / 1e6f
                   : (PROXY_WINDOW_MS / 1000.0f);

        bool enough_data = acc.count >= static_cast<uint32_t>(CURRENT_SAMPLE_HZ * 0.7f);
        bool all_ok = enough_data
                   && s_sensor.current_ok
                   && s_sensor.temperature_ok
                   && s_sensor.irradiance_ok
                   && std::isfinite(filtered_temp)
                   && std::isfinite(filtered_lux);

        if (all_ok && std::isfinite(prev_temp)) {
            s_proxy = compute_proxy_result(
                acc.sum,
                acc.sum_sq,
                acc.count,
                filtered_temp,
                prev_temp,
                dt_s,
                s_sensor.irradiance_wm2);

            apply_local_protection(s_proxy.status);

            ESP_LOGI(TAG,
                     "status=%s score=%.3f | I=%.3fA ripple=%.4f | "
                     "T=%.2fC dT/dt=%.4f | Lux=%.1f Irr=%.2f eta_dev=%.4f",
                     status_to_string(s_proxy.status),
                     s_proxy.score,
                     s_proxy.current_mean_a,
                     s_proxy.ripple,
                     s_sensor.temperature_c,
                     s_proxy.temp_rate_c_per_s,
                     s_sensor.illuminance_lux,
                     s_sensor.irradiance_wm2,
                     s_proxy.eta_dev);

#if DASNET_ENABLE_LORA
            char packet[255]{};
            size_t len = build_status_packet(packet, sizeof(packet), s_sensor, s_proxy);
            if (len > 0) {
                ++s_packet_seq;
                esp_err_t tx = lora_send(reinterpret_cast<const uint8_t *>(packet), len);
                if (tx != ESP_OK) {
                    // Failsafe: keputusan proteksi lokal TIDAK bergantung LoRa.
                    ESP_LOGW(TAG, "LoRa TX gagal: %s", esp_err_to_name(tx));
                }
            }
#endif
        } else {
            if (!first) {
                s_proxy.status = DasnetStatus::ERROR;
                apply_local_protection(DasnetStatus::ERROR);
                ESP_LOGW(TAG,
                         "Data sensor belum valid: current=%d temp=%d irradiance=%d samples=%lu",
                         s_sensor.current_ok,
                         s_sensor.temperature_ok,
                         s_sensor.irradiance_ok,
                         static_cast<unsigned long>(acc.count));
            }
        }

        if (temp_ok) {
            prev_temp = filtered_temp;
            prev_temp_time_us = now_us;
        }

        first = false;

        // Pertahankan perioda minimal ~1 s per evaluasi.
        int64_t elapsed_ms = (esp_timer_get_time() - cycle_start) / 1000;
        if (elapsed_ms < PROXY_WINDOW_MS) {
            vTaskDelay(pdMS_TO_TICKS(PROXY_WINDOW_MS - elapsed_ms));
        } else {
            taskYIELD();
        }
    }
}

static void indicator_task(void *arg) {
    (void)arg;

    while (true) {
        DasnetStatus status = s_proxy.status;

        if (status == DasnetStatus::WASPADA) {
            gpio_set_level(PIN_BUZZER, BUZZER_ON_LEVEL);
            vTaskDelay(pdMS_TO_TICKS(100));
            gpio_set_level(PIN_BUZZER, BUZZER_OFF_LEVEL);
            vTaskDelay(pdMS_TO_TICKS(900));
        } else if (status == DasnetStatus::BAHAYA) {
            // Bahaya: buzzer kontinu.
            gpio_set_level(PIN_BUZZER, BUZZER_ON_LEVEL);
            vTaskDelay(pdMS_TO_TICKS(250));
        } else {
            gpio_set_level(PIN_BUZZER, BUZZER_OFF_LEVEL);
            vTaskDelay(pdMS_TO_TICKS(250));
        }
    }
}

// ============================================================================
// 13. APP MAIN
// ============================================================================

extern "C" void app_main(void) {
    ESP_LOGI(TAG, "====================================================");
    ESP_LOGI(TAG, "DASNet Node %u | Tim Xtal | FW %s",
             DASNET_NODE_ID, DASNET_FW_VERSION);
    ESP_LOGI(TAG, "ESP32-S3 + FreeRTOS | Proxy Fusion + Edge AI + LoRa");
    ESP_LOGI(TAG, "====================================================");

#if !DASNET_ENABLE_RELAY_TRIP
    ESP_LOGW(TAG,
             "RELAY AUTO-TRIP DISABLED (safety default). "
             "Set DASNET_ENABLE_RELAY_TRIP=1 only after HIL validation.");
#endif

    ESP_ERROR_CHECK(init_outputs());

    esp_err_t adc_err = init_acs712_adc();
    if (adc_err != ESP_OK) {
        ESP_LOGE(TAG, "ACS712 init gagal: %s", esp_err_to_name(adc_err));
    }

    esp_err_t ds_err = init_ds18b20();
    if (ds_err != ESP_OK) {
        ESP_LOGW(TAG, "DS18B20 init warning: %s", esp_err_to_name(ds_err));
    }

    esp_err_t bh_err = init_i2c_and_bh1750();
    if (bh_err != ESP_OK) {
        ESP_LOGE(TAG, "BH1750 init gagal: %s", esp_err_to_name(bh_err));
    }

#if DASNET_ENABLE_LORA
    esp_err_t lora_err = init_lora();
    if (lora_err != ESP_OK) {
        ESP_LOGE(TAG, "LoRa init gagal: %s. Proteksi lokal tetap berjalan.",
                 esp_err_to_name(lora_err));
    }
#endif

#if DASNET_ENABLE_TINYML
    esp_err_t ml_err = init_tinyml();
    if (ml_err == ESP_OK) {
        tinyml_boot_benchmark();
    } else {
        ESP_LOGE(TAG, "TinyML init gagal: %s. Rule-based proxy tetap berjalan.",
                 esp_err_to_name(ml_err));
    }
#endif

    // State awal.
    s_proxy.status = DasnetStatus::ERROR;
    apply_local_protection(DasnetStatus::ERROR);

    BaseType_t ok1 = xTaskCreatePinnedToCore(
        current_sampling_task,
        "dasnet_current",
        4096,
        nullptr,
        5,
        nullptr,
        0);

    BaseType_t ok2 = xTaskCreatePinnedToCore(
        proxy_and_protection_task,
        "dasnet_proxy",
        8192,
        nullptr,
        4,
        nullptr,
        1);

    BaseType_t ok3 = xTaskCreatePinnedToCore(
        indicator_task,
        "dasnet_indicator",
        3072,
        nullptr,
        2,
        nullptr,
        1);

    if (ok1 != pdPASS || ok2 != pdPASS || ok3 != pdPASS) {
        ESP_LOGE(TAG, "Gagal membuat satu/lebih task FreeRTOS");
        apply_local_protection(DasnetStatus::ERROR);
    }

    ESP_LOGI(TAG, "DASNet firmware started.");
}
