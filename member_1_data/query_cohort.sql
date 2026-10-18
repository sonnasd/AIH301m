WITH cohort_base AS (
    SELECT ie.subject_id, ie.hadm_id, ie.icustay_id, ie.intime, ie.outtime,
           ROUND((EXTRACT(EPOCH FROM (ie.intime - pat.dob)) / (60*60*24*365.242))::numeric, 2) AS age,
           pat.gender, adm.hospital_expire_flag
    FROM icustays ie
    JOIN admissions adm ON ie.hadm_id = adm.hadm_id
    JOIN patients pat ON ie.subject_id = pat.subject_id
    WHERE EXTRACT(EPOCH FROM (ie.intime - pat.dob)) / (60*60*24*365.242) >= 18
),
abx_events AS (
    SELECT subject_id, hadm_id, startdate
    FROM prescriptions
    WHERE LOWER(drug) SIMILAR TO '%(vancomycin|ceftriaxone|piperacillin|tazobactam|meropenem|cefepime|ciprofloxacin|levofloxacin|ampicillin|metronidazole)%'
),
suspected_infection AS (
    SELECT abx.subject_id, abx.hadm_id,
           LEAST(abx.startdate, mic.charttime) AS suspected_infection_time
    FROM abx_events abx
    JOIN microbiologyevents mic ON abx.subject_id = mic.subject_id AND abx.hadm_id = mic.hadm_id
    WHERE mic.charttime >= abx.startdate - INTERVAL '72 HOURS'
      AND mic.charttime <= abx.startdate + INTERVAL '24 HOURS'
)
SELECT c.subject_id, c.hadm_id, c.icustay_id, c.age, c.gender,
       c.hospital_expire_flag, MIN(s.suspected_infection_time) AS suspected_infection_time,
       'suspected infection candidate only; SOFA not evaluated' AS cohort_scope
FROM cohort_base c
JOIN suspected_infection s ON c.subject_id = s.subject_id AND c.hadm_id = s.hadm_id
WHERE s.suspected_infection_time >= c.intime - INTERVAL '24 HOURS'
  AND s.suspected_infection_time <= c.outtime
GROUP BY c.subject_id, c.hadm_id, c.icustay_id, c.age, c.gender, c.hospital_expire_flag;
