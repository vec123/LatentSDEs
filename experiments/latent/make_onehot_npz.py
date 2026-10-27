"""
hs_sequences_all_light.npz (the arrays of hs_sequences_train.npz) with the categorical columns as one-hot columns:
visit_type, ethnicity, smoking_status, family_history_hs. A value of the column becomes one 0/1 column "<name>=<value>" (the
values that occur); a row where the column is missing (mask 0) has all its one-hot columns 0 with mask 0. The one-hot columns of
a static column stay static. The ordinal columns (hurley_stage_baseline, max_bio_line) and the binary sex stay as they are.
The original npz is not changed.

    python -I -m experiments.latent.make_onehot_npz
"""
import numpy as np

SRC = "test_data/hs_public_dataset/hs_sequences_all_light.npz"
DST = "test_data/hs_public_dataset/hs_sequences_onehot_light.npz"
CATEGORICAL = ["visit_type", "ethnicity", "smoking_status", "family_history_hs"]


def main():
    d = dict(np.load(SRC))
    names, static = list(d["feature_names"]), d["static_feature"]
    X, M = d["X"], d["M"]
    keep = [j for j, n in enumerate(names) if n not in CATEGORICAL]
    cols, masks, new_names, new_static = [X[:, keep]], [M[:, keep]], [names[j] for j in keep], [static[j] for j in keep]
    for n in CATEGORICAL:
        j = names.index(n)
        obs = M[:, j] > 0
        values = np.unique(X[obs, j])
        print(n, "values", values, "missing share", 1 - obs.mean())
        for v in values:
            cols.append(((X[:, j] == v) & obs).astype(np.float32)[:, None])
            masks.append(M[:, j:j + 1])
            new_names.append(f"{n}={int(v)}")
            new_static.append(static[j])
    d["X"], d["M"] = np.concatenate(cols, 1).astype(np.float32), np.concatenate(masks, 1).astype(M.dtype)
    d["feature_names"], d["static_feature"] = np.array(new_names), np.array(new_static)
    np.savez(DST, **d)
    print(f"{X.shape[1]} -> {d['X'].shape[1]} columns, saved {DST}")


if __name__ == "__main__":
    main()
