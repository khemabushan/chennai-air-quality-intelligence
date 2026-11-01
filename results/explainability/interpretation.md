**Strongest features on the test period (mean |SHAP|, PM2.5 units):** `pm10_t` (16.7), `pm25_t` (15.4), `pm10_roll7_mean` (7.1), `pm25_roll3_mean` (3.6), `pm25_roll7_mean` (2.4).

* **Recent PM2.5 history** (today's value, lags, rolling means/std) carries **45%** of total attribution — tomorrow's level is mostly anchored on the last few days, which is why persistence is already a strong baseline.
* **Same-day co-pollutants** (PM10, NO₂, SO₂, O₃) carry **46%**, dominated by PM10. PM10 and PM2.5 share emission sources and weather drivers, so PM10 adds information about the current particulate regime beyond PM2.5 alone.
* **Station identity** carries **5%**: the model learns station-specific offsets (e.g. Alandur and Kodungaiyur levels differ systematically).
* Calendar features contribute little at a one-day horizon once recent history is known.

Gain importance (how much each feature reduced training loss) ranks the same group of features highest, with `pm25_roll3_mean` and `pm25_t` on top. These are associations learned by the model, not causal effects.