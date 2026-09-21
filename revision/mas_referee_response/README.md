# MAS referee-response variants

This folder contains a standalone implementation of the MAS checks requested by the referees. It does not import or modify the original paper code in `Code/V002/iml_rev.py`.

## Inputs

The script reads the latest saved replication artifacts:

```text
Results/Updated CPI 3/shapleys_h1_upd.bin
Results/Updated CPI 3/shapleys_h3_upd.bin
Results/Updated CPI 3/shapleys_h6_upd.bin
Results/Updated CPI 3/shapleys_h12_upd.bin
```

Each artifact must contain:

```python
(
    (pbsv_rmse, pbsv_rmse_years),
    (oshapley, ishapley),
    oshapley_vi,
    ishapley_vi,
)
```

## Comparisons

The default run computes three appendix panels:

- Current MAS: in-sample TS-Shapley VI vs out-of-sample GPBSV.
- Referee check 1: in-sample TS-Shapley VI vs out-of-sample Shapley VI.
- Referee check 2: out-of-sample Shapley VI vs out-of-sample GPBSV.

The optional `--include-sensitivity` flag also computes out-of-sample Shapley VI vs out-of-sample GPBSV using in-sample Shapley weights.

## Usage

Quick validation run:

```powershell
.venv\Scripts\python.exe revision\mas_referee_response\mas_referee_variants.py --n-sims 1000
```

Full appendix run:

```powershell
.venv\Scripts\python.exe revision\mas_referee_response\mas_referee_variants.py --n-sims 1000000
```

Default outputs are isolated under:

```text
revision/mas_referee_response/outputs/
```

The script writes:

- `mas_referee_variants_details_a0.67_mc1000000.csv`
- `mas_referee_variants_a0.67_mc1000000.csv`
- `mas_referee_variants_pvalues_a0.67_mc1000000.csv`
- `mas_referee_variants_formatted_a0.67_mc1000000.csv`
- `mas_referee_variants_a0.67_mc1000000.xlsx`
- `mas_referee_variants_a0.67_mc1000000.tex`

Use `--output-dir` if the outputs should instead be written under `Results/Updated CPI 3/MAS/referee_variants/`.
