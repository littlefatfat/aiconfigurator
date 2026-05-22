# Sync refactor_test_aic from dash to H20

Apply on H20 from repo root `/data2/maruiyan.mry/aiconfigurator/`.

## Order
1. `0003-add-config_gb300_dsv4pro_tp4.patch` - new archived gb300 config
2. `0004-add-config_h20_dsv4flash_tp8_fp8.patch` - new archived h20-flash config
3. `0005-replace-config.py-with-gb300-dsv4pro.patch` - swap active config.py from flash -> gb300/dsv4pro (the original flash content is preserved by patch 0004 above)
4. `0006-stage2-predictor-overrides.patch` - scenario-aware floor + scale knobs

## Commands
```
cd /data2/maruiyan.mry/aiconfigurator
# dry-run all first
for p in /path/to/0003-*.patch /path/to/0004-*.patch /path/to/0005-*.patch /path/to/0006-*.patch; do
  echo "=== $p ==="
  patch -p1 --dry-run < "$p"
done
# apply
for p in /path/to/0003-*.patch /path/to/0004-*.patch /path/to/0005-*.patch /path/to/0006-*.patch; do
  patch -p1 < "$p"
done
```

Revert any single patch with `patch -p1 -R < <patch>`.

## After apply, verify
```
cd /data2/maruiyan.mry/aiconfigurator
git status refactor_test_aic/
python3 -c "from refactor_test_aic import config; print(config.AIC_SYSTEM, config.AIC_VERSION, config.MODEL_CONFIG_KWARGS['tp_size'])"
# expect: gb300 0.5.11_660b28976_dsv4pro 4
```

If you want to switch back to dsv4-flash for an H20 run, copy
`config_h20_dsv4flash_tp8_fp8.py` over `config.py`.
