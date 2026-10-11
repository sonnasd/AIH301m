WITH cohort_base AS (
    SELECT 
        ie.subject_id,
        ie.hadm_id,
        ie.icustay_id,
        ie.intime,
        ie.outtime,
        ROUND((EXTRACT(EPOCH FROM (ie.intime - pat.dob)) / (60*60*24*365.242)):[:NUMERIC], 2) AS age,
        pat.gender,
        adm.hospital_expire_flag,
        pat.dod
    FROM icustays ie
    INNER JOIN admissions adm ON ie.hadm_id = adm.hadm_id
    INNER JOIN patients pat ON ie.subject_id = pat.subject_id
    WHERE (EXTRACT(EPOCH FROM (ie.intime - pat.dob)) / (60*60*24*365.242)) >= 18
),
micro_events AS (
    SELECT 
        subject_id,
        hadm_id,
        chartdate,
        charttime,
        spec_type_desc
    FROM microbiologyevents
    WHERE spec_type_desc IN ('BLOOD CULTURE', 'URINE', 'SPUTUM', 'SWAB')
),
abx_events AS (
    SELECT 
        subject_id,
        hadm_id,
        startdate,
        enddate,
        drug
    FROM prescriptions
    WHERE LOWER(drug) SIMILAR TO '%(vancomycin|ceftriaxone|piperacillin|tazobactam|meropenem|cefepime|ciprofloxacin|levofloxacin|ampicillin|metronidazole)%'
),
sepsis_suspect AS (
    SELECT 
        abx.subject_id,
        abx.hadm_id,
        LEAST(abx.startdate, mic.charttime) AS sepsis_onset_time
    FROM abx_events abx
    INNER JOIN micro_events mic ON abx.hadm_id = mic.hadm_id
    WHERE mic.charttime >= abx.startdate - INTERVAL '24 HOURS'
      AND mic.charttime <= abx.startdate + INTERVAL '72 HOURS'
),
fluid_intake_4h AS (
    SELECT 
        icustay_id,
        DATE_TRUNC('hour', charttime) AS chart_hour,
        SUM(amount) AS total_fluid_ml
    FROM (
        SELECT icustay_id, charttime, amount FROM inputevents_mv WHERE amount > 0
        UNION ALL
        SELECT icustay_id, charttime, amount FROM inputevents_cv WHERE amount > 0
    ) f_all
    GROUP BY icustay_id, chart_hour
),
vaso_4h AS (
    SELECT 
        icustay_id,
        DATE_TRUNC('hour', charttime) AS chart_hour,
        MAX(rate) AS max_vaso_rate
    FROM (
        SELECT icustay_id, charttime, rate FROM inputevents_mv 
        WHERE itemid IN (221906, 221289, 221653, 221749) AND rate > 0
        UNION ALL
        SELECT icustay_id, charttime, rate FROM inputevents_cv 
        WHERE itemid IN (30047, 30120, 30044, 30119) AND rate > 0
    ) v_all
    GROUP BY icustay_id, chart_hour
)
SELECT 
    c.subject_id,
    c.hadm_id,
    c.icustay_id,
    c.age,
    c.gender,
    c.hospital_expire_flag,
    s.sepsis_onset_time
FROM cohort_base c
INNER JOIN sepsis_suspect s ON c.hadm_id = s.hadm_id;
