import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import { ROOT } from './layout.mjs';

export const read = path => fs.readFile(`${ROOT}/${path}`, 'utf8');
export const json = async path => JSON.parse(await read(path));
export async function csv(path) {
  const lines = (await read(path)).trim().split(/\r?\n/);
  const headers = lines.shift().split(',');
  return lines.map(line => Object.fromEntries(line.split(',').map((v, i) => [headers[i], v])));
}
export const policy = await json('artifacts/ai_policy.json');
export const metrics = await json('artifacts/evaluation_metrics.json');
export const cohort = await json('artifacts/cohort_manifest.json');
export const mortality = await csv('figures/mortality_bin_summary.csv');
export const sensitivity = await csv('data/sensitivity_analysis.csv');
export const splits = await Promise.all(['train', 'val', 'test'].map(async name => {
  const records = await csv(`data/trajectories_${name}.csv`);
  const patients = new Set(records.map(r => r.subject_id));
  const stays = new Set(records.map(r => [r.subject_id, r.hadm_id, r.icustay_id].join('/')));
  const terminal = records.filter(r => /true|1/i.test(r.done));
  const episodeReturns = new Map();
  for (const r of records) {
    const key = [r.subject_id, r.hadm_id, r.icustay_id].join('/');
    episodeReturns.set(key, (episodeReturns.get(key) ?? 0) + policy.gamma ** Number(r.time_step) * Number(r.reward));
  }
  return { name, rows: records.length, patients: patients.size, episodes: stays.size, terminal: terminal.length, episodeReturns };
}));
export const fmt = (v, digits = 2) => Number(v).toLocaleString('vi-VN', { minimumFractionDigits: digits, maximumFractionDigits: digits });
assert.deepEqual(splits.map(s => s.patients), [420, 90, 90]);
assert.deepEqual(splits.map(s => s.rows), [5496, 1173, 1171]);
assert.deepEqual(policy.history.map(x => x.actions_changed), [53, 8, 0]);
assert.equal(cohort.generation_parameters.features.length, 30);
assert.equal(policy.below_threshold_fallback_states.length, 5);
assert.equal(metrics.n_episodes, splits[2].episodes);
assert.equal(metrics.data_source, 'synthetic');
assert.equal(sensitivity.every(r => r.split === 'validation'), true);
assert.ok(Math.abs(0.5 * 0.99 * 60 + 0.5 * 100 - 79.7) < 1e-12);
assert.ok(Math.abs(50 / (1 - 0.495) - 99.00990099009901) < 1e-10);
