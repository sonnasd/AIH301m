import numpy as np
import pandas as pd

def generate_sepsis_cohort(n_patients=600, max_steps=18, random_state=42):
    if not isinstance(n_patients, int) or n_patients < 1:
        raise ValueError("n_patients must be a positive integer")
    if not isinstance(max_steps, int) or max_steps < 8:
        raise ValueError("max_steps must be an integer >= 8")
    rng = np.random.RandomState(random_state)
    records = []
    
    for i in range(n_patients):
        subject_id = 10000 + i
        hadm_id = 150000 + i
        icustay_id = 200000 + i
        
        age = int(rng.normal(65, 12))
        age = max(18, min(age, 90))
        gender = int(rng.choice([0, 1], p=[0.45, 0.55]))
        elixhauser = int(np.clip(rng.poisson(3), 0, 10))
        initial_sofa = int(np.clip(rng.poisson(6), 2, 18))
        
        n_steps = rng.randint(8, max_steps + 1)
        
        gcs = rng.choice([15, 14, 13, 11, 8], p=[0.55, 0.20, 0.12, 0.08, 0.05])
        hr = rng.normal(88, 12)
        sys_bp = rng.normal(110, 15)
        dia_bp = rng.normal(62, 10)
        map_val = (sys_bp + 2 * dia_bp) / 3.0
        rr = rng.normal(20, 4)
        temp = rng.normal(37.5, 0.8)
        spo2 = rng.normal(96, 3)
        fio2 = rng.choice([0.21, 0.40, 0.50, 0.60, 0.80], p=[0.40, 0.30, 0.15, 0.10, 0.05])
        
        lactate = rng.exponential(1.8) + 0.8
        creatinine = rng.exponential(1.2) + 0.7
        platelets = rng.normal(220, 60)
        bilirubin = rng.exponential(1.0) + 0.4
        wbc = rng.normal(12.5, 4.5)
        bun = rng.normal(25, 10)
        glucose = rng.normal(135, 35)
        potassium = rng.normal(4.2, 0.5)
        sodium = rng.normal(138, 4)
        chloride = rng.normal(102, 5)
        bicarbonate = rng.normal(22, 4)
        ph = rng.normal(7.34, 0.06)
        pao2 = rng.normal(95, 20)
        paco2 = rng.normal(40, 6)
        hemoglobin = rng.normal(10.5, 1.8)
        
        urine = rng.normal(180, 60)
        cum_balance = rng.normal(500, 300)
        
        patient_trajectory = []
        
        for step in range(n_steps):
            hr = float(np.clip(hr + rng.normal(0, 3), 45, 160))
            sys_bp = float(np.clip(sys_bp + rng.normal(0, 5), 55, 190))
            dia_bp = float(np.clip(dia_bp + rng.normal(0, 4), 30, 110))
            map_val = float((sys_bp + 2 * dia_bp) / 3.0)
            rr = float(np.clip(rr + rng.normal(0, 2), 8, 42))
            temp = float(np.clip(temp + rng.normal(0, 0.2), 35.0, 40.5))
            spo2 = float(np.clip(spo2 + rng.normal(0, 1.5), 75, 100))
            
            lactate = float(np.clip(lactate + rng.normal(0, 0.25), 0.5, 15.0))
            creatinine = float(np.clip(creatinine + rng.normal(0, 0.12), 0.3, 10.0))
            platelets = float(np.clip(platelets + rng.normal(0, 8), 15, 600))
            bilirubin = float(np.clip(bilirubin + rng.normal(0, 0.08), 0.1, 15.0))
            wbc = float(np.clip(wbc + rng.normal(0, 0.7), 1.0, 40.0))
            bun = float(np.clip(bun + rng.normal(0, 1.5), 5.0, 120.0))
            glucose = float(np.clip(glucose + rng.normal(0, 6.0), 40.0, 450.0))
            potassium = float(np.clip(potassium + rng.normal(0, 0.1), 2.2, 7.0))
            sodium = float(np.clip(sodium + rng.normal(0, 0.8), 115, 160))
            chloride = float(np.clip(chloride + rng.normal(0, 0.8), 80, 130))
            bicarbonate = float(np.clip(bicarbonate + rng.normal(0, 0.6), 10, 45))
            ph = float(np.clip(ph + rng.normal(0, 0.02), 6.9, 7.6))
            pao2 = float(np.clip(pao2 + rng.normal(0, 4.0), 45, 300))
            paco2 = float(np.clip(paco2 + rng.normal(0, 1.5), 18, 85))
            hemoglobin = float(np.clip(hemoglobin + rng.normal(0, 0.2), 4.5, 18.0))
            
            urine = float(np.clip(urine + rng.normal(0, 20), 0, 700))
            
            prob_zero_fluid = 0.35 if map_val > 70 else 0.10
            if rng.rand() < prob_zero_fluid:
                iv_fluids = 0.0
            else:
                iv_fluids = float(np.clip(rng.exponential(250) + 50, 10, 2000))
                
            prob_zero_vaso = 0.70 if map_val > 65 else 0.25
            if rng.rand() < prob_zero_vaso:
                vaso = 0.0
            else:
                vaso = float(np.clip(rng.exponential(0.18) + 0.02, 0.01, 1.8))
                
            row = {
                'subject_id': subject_id,
                'hadm_id': hadm_id,
                'icustay_id': icustay_id,
                'step': step,
                'age': age,
                'gender': gender,
                'elixhauser': elixhauser,
                'sofa_initial': initial_sofa,
                'gcs': gcs,
                'hr': hr,
                'sys_bp': sys_bp,
                'dia_bp': dia_bp,
                'map': map_val,
                'rr': rr,
                'temp': temp,
                'spo2': spo2,
                'fio2': fio2,
                'lactate': lactate,
                'creatinine': creatinine,
                'platelets': platelets,
                'bilirubin': bilirubin,
                'wbc': wbc,
                'bun': bun,
                'glucose': glucose,
                'potassium': potassium,
                'sodium': sodium,
                'chloride': chloride,
                'bicarbonate': bicarbonate,
                'ph': ph,
                'pao2': pao2,
                'paco2': paco2,
                'hemoglobin': hemoglobin,
                'urine_output': urine,
                'cumulated_balance': cum_balance,
                'iv_fluids': iv_fluids,
                'vasopressors': vaso
            }
            patient_trajectory.append(row)
            pressure_gain = 0.0
            if iv_fluids > 0:
                pressure_gain += rng.uniform(1.0, 4.0)
            if vaso > 0:
                pressure_gain += rng.uniform(3.0, 8.0)
            sys_bp = float(np.clip(sys_bp + pressure_gain, 55, 190))
            dia_bp = float(np.clip(dia_bp + pressure_gain, 30, 110))
            map_val = (sys_bp + 2 * dia_bp) / 3.0
            cum_balance += iv_fluids - urine
            
        final_state = patient_trajectory[-1]
        mortality_logit = (
            -3.4
            + 0.03 * (final_state['age'] - 65)
            + 0.28 * (final_state['lactate'] - 2.0)
            - 0.05 * (map_val - 65)
            + 0.16 * (final_state['sofa_initial'] - 6)
            + 0.10 * (final_state['creatinine'] - 1.2)
        )
        mortality_prob = 1.0 / (1.0 + np.exp(-mortality_logit))
        mortality_prob = float(np.clip(mortality_prob, 0.03, 0.85))
        died = int(rng.rand() < mortality_prob)
        
        for record in patient_trajectory:
            record['died'] = died
            records.append(record)
            
    df = pd.DataFrame(records)
    return df
