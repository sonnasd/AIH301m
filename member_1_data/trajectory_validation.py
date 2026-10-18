import numpy as np


def validate_trajectories(df, n_states, n_actions):
    required = ['subject_id', 'time_step', 'state', 'action', 'next_state', 'reward', 'done']
    missing = set(required) - set(df.columns)
    if missing or df.empty:
        raise ValueError(f"Empty trajectories or missing columns: {sorted(missing)}")
    episode_cols = ['subject_id'] + [c for c in ['hadm_id', 'icustay_id'] if c in df]
    if df[episode_cols].isna().any().any():
        raise ValueError("Missing episode identifiers")
    if not np.isfinite(df[required[1:]].to_numpy(dtype=float)).all():
        raise ValueError("Trajectory values must be finite")
    for col, limit in [('state', n_states), ('action', n_actions), ('next_state', n_states + 2)]:
        if not ((df[col] >= 0) & (df[col] < limit) & (df[col] % 1 == 0)).all():
            raise ValueError(f"Invalid {col}")
    if not df['done'].isin([True, False]).all():
        raise ValueError("done must be boolean")
    for _, group in df.groupby(episode_cols, sort=False):
        ordered = group.sort_values('time_step')
        steps = ordered['time_step'].to_numpy()
        if (steps < 0).any() or (steps % 1 != 0).any() or (np.diff(steps) != 1).any():
            raise ValueError("Episode steps must be unique contiguous nonnegative integers")
        done = ordered['done'].to_numpy(dtype=bool)
        if not done[-1] or done[:-1].any():
            raise ValueError("Each episode must have exactly one final terminal transition")
        states = ordered['state'].to_numpy(dtype=int)
        next_states = ordered['next_state'].to_numpy(dtype=int)
        if not np.array_equal(next_states[:-1], states[1:]) or next_states[-1] < n_states:
            raise ValueError("Broken transition chain or invalid terminal state")
    return episode_cols
