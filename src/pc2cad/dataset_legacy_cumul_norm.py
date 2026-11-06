"""Pre-normalized single-shot dataset.

The new parquet corpus stores shapes already in [-1, 1]^3, so this
dataset skips per-sample bbox normalization entirely and passes a zero
geom_cond token. Everything else (reader, float-bin collate, sampler)
matches dataset_legacy_cumul.

With the 12M-row corpus, the per-rank init scan (reading index columns
plus cumulative_code for max_target_floats filter) takes 30+ min * 8
ranks on NVMe. We avoid this with:
  - pairs cache (keyed on parquet_dir+folders) written by whichever rank
    grabs a file-lock first; other ranks load instantly
  - max_target_floats filter moved to collate-time (zero init cost)
"""

import os, re, time, glob, functools, hashlib, fcntl
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
import pyarrow.parquet as pq
import pyarrow as pa
from pc2cad.model_legacy_norm import float_to_bin, make_float_token, FLOAT_MAX_VAL


_CACHE_DIR = "/tmp/pc2cad_dataset_cache"
os.makedirs(_CACHE_DIR, exist_ok=True)

FLOAT_RE = re.compile(r"\d+\.\d+")


_INDEX_COLUMNS = ["folder", "model", "op_index", "num_ops"]
_FILTER_COLUMNS = ["volume", "bbox_min", "bbox_max"]
_READ_COLUMNS = ["pointcloud", "pc_num_points", "pc_dims",
                 "cumulative_code", "code", "normals"]


class ParquetRowReader:
    def __init__(self, paths, cache_size=4):
        t0 = time.time()
        self._index = []
        self._cum_rows = [0]
        total = 0
        for p in paths:
            m = pq.read_metadata(p)
            for rg in range(m.num_row_groups):
                n = m.row_group(rg).num_rows
                self._index.append((p, rg, n))
                total += n
                self._cum_rows.append(total)
        self._len = total
        self._paths = paths
        self._cache_size = cache_size
        self._cache = None
        print(f"  ParquetRowReader (norm): {total:,} rows in {len(paths)} files "
              f"({time.time()-t0:.1f}s)", flush=True)

    def _init_cache(self):
        @functools.lru_cache(maxsize=self._cache_size)
        def _r(path, rg):
            pf = pq.ParquetFile(path)
            return pf.read_row_group(rg, columns=_READ_COLUMNS)
        self._cache = _r

    def __len__(self):
        return self._len

    def _locate(self, gi):
        lo, hi = 0, len(self._index) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self._cum_rows[mid + 1] <= gi:
                lo = mid + 1
            else:
                hi = mid
        p, rg, _ = self._index[lo]
        return p, rg, gi - self._cum_rows[lo]

    def read_row(self, gi):
        if self._cache is None:
            self._init_cache()
        path, rg, local = self._locate(gi)
        t = self._cache(path, rg)
        r = t.slice(local, 1)
        return {c: r.column(c)[0].as_py() for c in r.column_names}

    def read_columns(self, columns):
        t0 = time.time()
        ts = [pq.read_table(p, columns=columns) for p in self._paths]
        t = pa.concat_tables(ts)
        out = {c: t.column(c).to_pylist() for c in columns}
        print(f"  Read {columns}: {len(out[columns[0]]):,} rows "
              f"({time.time()-t0:.1f}s)", flush=True)
        return out

    def row_group_for_idx(self, gi):
        lo, hi = 0, len(self._index) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self._cum_rows[mid + 1] <= gi:
                lo = mid + 1
            else:
                hi = mid
        return lo


def _equalize_size(pc, target):
    """Random-uniform subsample/upsample to `target` rows."""
    n = pc.shape[0]
    if target is None or n == target:
        return pc
    if n > target:
        idx = np.random.choice(n, target, replace=False)
        return pc[idx]
    extra = np.random.choice(n, target - n, replace=True)
    return np.concatenate([pc, pc[extra]], axis=0)


def _fps_subsample(pc, target):
    """Farthest-point-sampling subsample of `pc` to `target` rows.
    Distance computed on first 3 dims (xyz). For pc.shape[0] <= target
    behaves like _equalize_size (with up-sampling fallback)."""
    n = pc.shape[0]
    if target is None or n == target:
        return pc
    if n < target:
        # Same up-sample fallback as random.
        extra = np.random.choice(n, target - n, replace=True)
        return np.concatenate([pc, pc[extra]], axis=0)
    xyz = pc[:, :3]
    out_idx = np.empty(target, dtype=np.int64)
    out_idx[0] = np.random.randint(n)
    dist = np.full(n, np.inf, dtype=np.float32)
    last = xyz[out_idx[0]]
    for i in range(1, target):
        d = ((xyz - last) ** 2).sum(-1)
        np.minimum(dist, d, out=dist)
        nxt = int(dist.argmax())
        out_idx[i] = nxt
        last = xyz[nxt]
    return pc[out_idx]


class PC2CADSingleShotNormDataset(Dataset):
    """Point clouds are already in [-1, 1]^3. No bbox normalization; no
    per-sample geom conditioning. One sample per (folder, model) pair
    (we take the final-op row).

    max_target_floats is applied AT COLLATE, not here — skipping the
    expensive cumulative_code read at init time. The pairs list itself
    is cached on disk (pickled per (parquet_dir, folders) key); the
    first rank to start populates the cache behind a file-lock, the
    rest load instantly.
    """

    def __init__(self, parquet_dir, folders, num_points=1024,
                 load_normals=False,
                 input_subsample="random",
                 max_target_floats=None,
                 min_volume=None, max_volume=None,
                 bbox_limit=None):
        """
        min_volume / max_volume: optional float thresholds on the parquet
          `volume` column of each model's final row. Samples with volume
          outside `[min_volume, max_volume]` are excluded from the pairs
          list at cache-build time.
        bbox_limit: optional float; excludes models whose bbox_min has
          any component < -bbox_limit or bbox_max any component > bbox_limit.
          Typical: 1.0 to enforce [-1, 1]^3 per-axis.
        """
        paths = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
        if not paths:
            paths = sorted(glob.glob(os.path.join(parquet_dir, "*/*.parquet")))
        paths = [p for p in paths
                 if any(f == os.path.basename(os.path.dirname(p))
                        or os.path.basename(p).split("_")[0] == f
                        for f in folders)]
        if not paths:
            raise ValueError(f"No parquets for folders {folders[:5]} in {parquet_dir}")
        print(f"  PC2CADSingleShotNormDataset: {len(paths)} files", flush=True)
        self.reader = ParquetRowReader(paths)
        self.num_points = num_points
        self.load_normals = load_normals
        assert input_subsample in ("random", "fps"), \
            f"input_subsample must be 'random' or 'fps', got {input_subsample}"
        self.input_subsample = input_subsample
        # Collate-time filter (free if None).
        self.max_target_floats = max_target_floats
        self.min_volume = min_volume
        self.max_volume = max_volume
        self.bbox_limit = bbox_limit

        # ---- Pairs cache ----
        # Key captures the sorted list of parquet paths AND the extreme
        # filter spec, so different filter configurations get different
        # caches.
        filt_spec = (
            f"mv={min_volume}|xv={max_volume}|bb={bbox_limit}")
        key_src = "\n".join(paths) + "\n" + filt_spec
        key = hashlib.md5(key_src.encode()).hexdigest()[:16]
        cache_file = os.path.join(_CACHE_DIR, f"pairs_{key}.npy")
        lock_file = cache_file + ".lock"

        any_filter = (min_volume is not None or max_volume is not None
                      or bbox_limit is not None)

        if os.path.exists(cache_file):
            self.pairs = np.load(cache_file).tolist()
            print(f"  loaded {len(self.pairs):,} pairs from cache {cache_file}",
                  flush=True)
        else:
            # Only one rank does the scan; others wait for the lock then
            # find the cache populated.
            with open(lock_file, "w") as lf:
                fcntl.flock(lf, fcntl.LOCK_EX)
                if os.path.exists(cache_file):
                    self.pairs = np.load(cache_file).tolist()
                    print(f"  (raced on lock) loaded {len(self.pairs):,} from cache",
                          flush=True)
                else:
                    print(f"  building pairs cache ({cache_file}) "
                          f"filt={filt_spec} ...", flush=True)
                    cols_to_read = list(_INDEX_COLUMNS)
                    if any_filter:
                        cols_to_read = cols_to_read + _FILTER_COLUMNS
                    cols = self.reader.read_columns(cols_to_read)
                    n = len(cols["folder"])
                    last_by_model = {}
                    n_drop_vol = 0; n_drop_bbox = 0
                    for i in range(n):
                        mkey = (cols["folder"][i], cols["model"][i])
                        if cols["op_index"][i] != cols["num_ops"][i] - 1:
                            continue
                        # Apply per-row extreme filter (on the final row).
                        if any_filter:
                            v = cols["volume"][i]
                            if v is None or (isinstance(v, float) and v != v):
                                n_drop_vol += 1; continue
                            if min_volume is not None and v < min_volume:
                                n_drop_vol += 1; continue
                            if max_volume is not None and v > max_volume:
                                n_drop_vol += 1; continue
                            if bbox_limit is not None:
                                bmn = cols["bbox_min"][i]
                                bmx = cols["bbox_max"][i]
                                if (min(bmn) < -bbox_limit
                                        or max(bmx) > bbox_limit):
                                    n_drop_bbox += 1; continue
                        last_by_model[mkey] = i
                    self.pairs = list(last_by_model.values())
                    np.save(cache_file, np.array(self.pairs, dtype=np.int64))
                    print(f"  cached {len(self.pairs):,} pairs -> {cache_file}",
                          flush=True)
                    if any_filter:
                        print(f"  dropped by filter: vol={n_drop_vol:,} "
                              f"bbox={n_drop_bbox:,}", flush=True)
                fcntl.flock(lf, fcntl.LOCK_UN)

        print(f"  {len(self.pairs):,} single-shot samples (pre-normalized)"
              + (f" [collate filter: max {max_target_floats} floats]"
                 if max_target_floats else "")
              + (f" [extreme filter: vol in [{min_volume},{max_volume}] "
                 f"bbox in [-{bbox_limit},{bbox_limit}]]" if any_filter else ""),
              flush=True)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        gi = self.pairs[idx]
        row = self.reader.read_row(gi)
        N = row["pc_num_points"]
        pc_xyz = np.frombuffer(row["pointcloud"], dtype=np.float32).copy()
        pc_xyz = pc_xyz.reshape(N, row["pc_dims"])[:, :3]
        if self.load_normals:
            nrm = np.frombuffer(row["normals"], dtype=np.float32).copy()
            nrm = nrm.reshape(N, 3)
            pc = np.concatenate([pc_xyz, nrm], axis=1)  # (N, 6)
        else:
            pc = pc_xyz
        # NO normalization — PC is already in [-1, 1]^3.
        if self.input_subsample == "fps":
            pc_sub = _fps_subsample(pc.copy(), self.num_points)
        else:
            pc_sub = _equalize_size(pc.copy(), self.num_points)
        # Zero geom_cond token. Shape/scale information is baked into PC.
        geom = np.zeros(4, dtype=np.float32)
        return {
            "pointcloud": torch.from_numpy(pc_sub).float(),
            "raw_pointcloud_full": torch.from_numpy(pc_xyz.copy()).float(),
            "geom_cond": torch.from_numpy(geom).float(),
            "code": row["cumulative_code"],
        }


class FileGroupedSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle=True, seed=42,
                 rank=0, world_size=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.rank = rank
        self.world_size = max(1, world_size)
        self.epoch = 0
        rg_to_indices = {}
        for i, gi in enumerate(dataset.pairs):
            rg = dataset.reader.row_group_for_idx(gi)
            rg_to_indices.setdefault(rg, []).append(i)
        self.groups = list(rg_to_indices.values())
        total = sum(len(g) for g in self.groups)
        self._per_rank_len = total // self.world_size

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch)
        groups = [g.copy() for g in self.groups]
        if self.shuffle:
            rng.shuffle(groups)
            for g in groups:
                rng.shuffle(g)
        shard = groups[self.rank::self.world_size]
        out = []
        for g in shard:
            out.extend(g)
        return iter(out[:self._per_rank_len])

    def __len__(self):
        return self._per_rank_len

    def set_epoch(self, e):
        self.epoch = e


def collate_fn(batch, tokenizer, num_points, pc_pad_id, geom_pad_id,
               max_seq_len, max_target_floats=None):
    """Drops samples with any |float| >= FLOAT_MAX_VAL, or (optionally)
    with more than `max_target_floats` numeric literals in the target.
    The max_target_floats check runs here at collate-time to avoid an
    expensive full-corpus cumulative_code read at dataset init."""
    pcs, geoms = [], []
    batch_ids, batch_labels = [], []
    eos_id = tokenizer.eos_token_id

    def _to_bin(text):
        def sub(m):
            v = float(m.group())
            if v >= FLOAT_MAX_VAL:
                return m.group()
            return make_float_token(float_to_bin(v))
        return FLOAT_RE.sub(sub, text)

    for sample in batch:
        code = sample["code"]
        floats = [float(m.group()) for m in FLOAT_RE.finditer(code)]
        if any(v >= FLOAT_MAX_VAL for v in floats):
            continue
        if max_target_floats is not None and len(floats) > max_target_floats:
            continue
        pcs.append(sample["pointcloud"])
        geoms.append(sample["geom_cond"])
        pc_ids = [pc_pad_id] * num_points
        geom_ids = [geom_pad_id]
        prompt = "# Generate the full CadQuery script for this shape:\n"
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        template = _to_bin(code) + "\n"
        target_ids = tokenizer.encode(template, add_special_tokens=False) + [eos_id]
        full_ids = pc_ids + geom_ids + prompt_ids + target_ids
        ctx = len(pc_ids) + len(geom_ids) + len(prompt_ids)
        full_labels = [-100] * ctx + target_ids
        if len(full_ids) > max_seq_len:
            full_ids = full_ids[:max_seq_len]
            full_labels = full_labels[:max_seq_len]
        batch_ids.append(full_ids)
        batch_labels.append(full_labels)

    if not batch_ids:
        return None

    L = max(len(x) for x in batch_ids)
    pad = tokenizer.pad_token_id
    pids, plabs, pmask = [], [], []
    for ids, labs in zip(batch_ids, batch_labels):
        n = L - len(ids)
        pids.append(ids + [pad] * n)
        plabs.append(labs + [-100] * n)
        pmask.append([1] * len(ids) + [0] * n)
    return {
        "input_ids": torch.tensor(pids, dtype=torch.long),
        "attention_mask": torch.tensor(pmask, dtype=torch.long),
        "labels": torch.tensor(plabs, dtype=torch.long),
        "pointcloud": torch.stack(pcs),
        "geom_cond": torch.stack(geoms),
    }
