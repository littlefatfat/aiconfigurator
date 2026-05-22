import json

CONFIG_PATH = "/host/insight_benchmark/test/hisim/mry_debug/one_case/l3.hisim_config.glm5.json"

with open(CONFIG_PATH) as f:
    c = json.load(f)

print(f"Before: data_type={c['scheduler']['data_type']}, kv_cache_data_type={c['scheduler']['kv_cache_data_type']}")

c["scheduler"]["data_type"] = "FP4"
c["scheduler"]["kv_cache_data_type"] = "FP8"

with open(CONFIG_PATH, "w") as f:
    json.dump(c, f, indent=4)

print(f"After:  data_type=FP4, kv_cache_data_type=FP8 - Fixed!")
