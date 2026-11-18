# Rule-based reference: all rules in their final form (rule set v1.4)

| Axis | Variable(s) | Rule | Threshold | State condition | Time unit |
|---|---|---|---|---|---|
| Static | Latitude, Longitude | admissible range | ±90°, ±180° | — | — |
| Static | HEADING, WIND ANGLE(R) | admissible range | 0 ≤ x < 360° | — | — |
| Static | RUDDER ANGLE | admissible range | ±40° | — | — |
| Static | SPEED(SOG) | admissible range | 0–27 kn | — | — |
| Static | SPEED(STW) | admissible range | −5 to 25.5 kn; below −1 kn underway | underway rule at SOG ≥ 5 kn | — |
| Static | FWD DRAFT | sensor rails | 2.0 < x < 17.9 m (the five ships of the second series: x < 14.5 m) | — | — |
| Static | AFT DRAFT | admissible range | 6.0–17.0 m | — | — |
| Static | MID(P) DRAFT, MID(S) DRAFT | admissible range | 5.5–16.65 m | — | — |
| Static | FORE DRAFT Noon, AFT DRAFT Noon | admissible range | 5.0–16.55 m, 8.0–16.55 m | — | — |
| Static | TRIM, Trim Noon | distribution gap cut-off | 1 % tails, gap ≥ 1.0 m | — | — |
| Static | ME LOAD, ME RPM, ME RPM(INT), ME SHAFT POWER | admissible range | 0–100 %, absolute value ≤ 80 rpm, 0–60,000 kW | — | — |
| Static | ME SHAFT TORQUE | admissible range | −500 to 7,500 kNm, after per-ship scale correction | — | — |
| Static | ME FO FLOW | admissible range | 0–12,000 L/h | — | — |
| Static | AE FO FLOW, AB FO FLOW | admissible range | 0–3,000, 0–2,000 L/h | — | — |
| Static | WIND SPEED(R), WIND SPEED(T) | admissible range | 0–40, 0–50 m/s | — | — |
| Static | WATER DEPTH-M | admissible range | > 0 m; ≥ 2 m underway | underway rule at SOG ≥ 5 kn | — |
| Static | ER TEMP | admissible range | 5–60 °C (one ship: ≤ 46 °C) | — | — |
| Static | SEA WATER TEMP, sst, swh, mwp | admissible range | 0.5–36 °C; sst ≥ 271.25 K; swh ≥ 0; mwp > 0 | — | — |
| Static | DEADWEIGHT | calibrated range | 60,000 t to the ship's observed maximum | — | — |
| Static | DISPLACEMENT | calibrated range | lightship + 60,000 t to lightship + deadweight maximum | — | — |
| Spike | FWD DRAFT, AFT DRAFT | deviation from a centred 31-row median above the ship's 99.9th percentile, with readings 5 rows before and after within 1.0 m | per ship: 2.24–2.77 m fwd, 0.64–0.91 m aft | underway | rows (nominal 1 min) |
| Rate of change | HEADING (circular), SPEED(SOG), WIND SPEED(R), WIND SPEED(T), SEA WATER TEMP, ER TEMP | absolute change per minute | 20°, 2.0 kn, 30 m/s, 15 m/s, 3.0 °C, 5.0 °C | — | minutes |
| Freezing | HEADING, SPEED(STW), MID(P)/MID(S) DRAFT, WIND ANGLE(R), WATER DEPTH-M | identical reading | ≥ 120 min | underway; heading rule off in months logged without decimals | minutes |
| Freezing | SPEED(SOG) | identical reading, checked against the speed from successive position fixes 1–5 min apart | ≥ 120 min and (position fixes on < 50 % of the run or \lvert SOG − position speed\rvert > 2 kn) | underway | minutes |
| Freezing | RUDDER ANGLE, ER TEMP, SEA WATER TEMP | identical reading | ≥ 120, 360, 720 min | underway | minutes |
| Freezing | ME LOAD, ME RPM, ME RPM(INT), ME SHAFT POWER, ME FO FLOW, WIND SPEED(R) | reading at or below a floor | ≤ 0 (≤ 1 for power and fuel flow) for ≥ 120 min | underway | minutes |
| Freezing | Latitude, Longitude | position displacement over 120 min | ≤ 200 m | SOG ≥ 5 kn | minutes |
| Relation | DISPLACEMENT − DEADWEIGHT | lightweight near its per-ship median | max(5,000 t, 8 %) | — | — |
| Relation | ME SHAFT POWER, ME RPM, ME SHAFT TORQUE | P = α · T · π · n / 30 | 50 % after per-ship α | revolutions ≥ 1 rpm in magnitude | — |
| Relation | TRIM, AFT DRAFT, FWD DRAFT | trim equals the draught difference | 1.0 m | — | — |
| Relation | SPEED(STW), SPEED(SOG) | difference | > 6 kn for ≥ 30 min, STW flagged | underway | minutes |
| Relation | ME FO FLOW, ME SHAFT POWER | daily median of FO/P from readings with P > 5,000 kW and 0 < FO ≤ 16,000 L/h, on days with ≥ 60 valid minutes | outside 0.13–0.26 L/kWh; gaps ≤ 30 days between flagged days also flagged; all FO readings of a flagged day marked | underway | days |
| Row flag | SPEED(SOG), position | berthed: 90 % of a 60-min window at SOG ≤ 0.5 kn and displacement ≤ 100 m | — | — | minutes |
| Normalisation | WIND ANGLE(T), mwd | reduced modulo 360°, not flagged | — | — | — |

Lightweight medians 63,002.9 t (seven ships) and 63,844.9 t (five ships); α = 0.9989–1.0002 on eleven ships (9.9999 on one before its torque scale was corrected by a factor of ten). Rules deferred for lack of an external grid: sea-water temperature against OISST, water depth against ETOPO. No rule: UTC, VESSELCODE, ShipName, DGPS EW/NS, ME SHAFT THRUST, u10, v10.

Ship-specific calibration values (continuation of Table S1):

| Ship |   Forward spike (m) |   Aft spike (m) |   Power factor α |   Lightweight (t) |   Deadweight upper limit (t) |
|:--------------|--------------------:|----------------:|-----------------:|------------------:|-----------------------------:|
| A |            2.506748 |        0.793000 |         1.000019 |         63002.906 |                   232606.094 |
| B |            2.454356 |        0.749000 |         0.999418 |         63002.906 |                   218061.094 |
| C |            2.497160 |        0.788000 |         1.000245 |         63002.906 |                   232606.094 |
| D |            2.700671 |        0.807000 |         0.999642 |         63002.906 |                   232606.500 |
| E |            2.766000 |        0.635712 |         0.998892 |         63002.906 |                   232606.094 |
| F |            2.377000 |        0.774000 |         0.999498 |         63002.906 |                   232606.094 |
| G |            2.242001 |        0.758000 |         0.999904 |         63002.906 |                   212800.297 |
| H |            2.676078 |        0.905000 |         0.999987 |         63844.906 |                   229015.094 |
| I |            2.595000 |        0.797999 |         0.999995 |         63844.906 |                   228379.594 |
| J |            2.523001 |        0.729000 |         0.999972 |         63844.906 |                   228379.594 |
| K |            2.435887 |        0.831000 |         0.999990 |         63844.906 |                   228379.594 |
| L |            2.443429 |        0.776000 |         0.999983 |         63844.906 |                   228379.594 |

| Ship | Sensor trim lower (m)   | Sensor trim upper (m)   | Noon trim lower (m)   | Noon trim upper (m)   |
|:--------------|:------------------------|:------------------------|:----------------------|:----------------------|
| A | -6.603                  | —                       | -1.9                  | —                     |
| B | -9.9005                 | —                       | —                     | 4.5                   |
| C | -8.946                  | 7.113                   | —                     | —                     |
| D | -9.119                  | —                       | —                     | —                     |
| E | —                       | —                       | —                     | —                     |
| F | —                       | 7.0565       | —                     | 3.25                  |
| G | —                       | 5.411                   | —                     | —                     |
| H | -10.2665                | 12.098                  | —                     | 4.45     |
| I | -9.805                  | —                       | —                     | 3.65                  |
| J | -8.1285      | 6.027                   | —                     | —                     |
| K | -15.587                 | 7.1905                  | -5.5                  | 3.825                 |
| L | -7.654                  | 8.3305                  | —                     | —                     |

A dash means that the tail-gap criterion found no cutoff on that side. The displacement limits are the listed lightweight plus 60,000 t and plus the deadweight upper limit. The lightweight tolerance is 8% of the median (greater than 5,000 t for every ship). The listed power factors are after correction of the torque of ship K by a factor of ten; the historical factor near ten preceded that correction. The heading-freezing guard disables the rule in months with fewer than 10% non-integer readings. The position-freezing check uses pairs of fixes 120 to 125 elapsed minutes apart.

For ship J, raw monthly fuel–power medians were checked using the rule's underway, power and fuel admissibility conditions, before its daily minimum-sample requirement. Medians range from 0.3385 to 0.3675 L/kWh in 27 observed months between August 2020 and December 2022. April–May 2022 and January 2023 have no eligible readings; February 2023 has only 13 (median 0.3658), and March has 527 (median 0.1788). These data establish a prolonged ratio anomaly, but cannot date a meter fault or correction precisely.
