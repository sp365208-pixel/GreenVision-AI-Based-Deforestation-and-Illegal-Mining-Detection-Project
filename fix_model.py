"""
fix_model.py - Run once to fix the corrupted model config
Usage: python backend/fix_model.py
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import json
import re

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_ressources", "att_unet_4b.hdf5")
BACKUP_PATH = MODEL_PATH + ".backup"

# ── Step 1: Make a backup if not already done ──────────────────
if not os.path.exists(BACKUP_PATH):
    import shutil
    shutil.copy2(MODEL_PATH, BACKUP_PATH)
    print(f"✅ Backup saved to: {BACKUP_PATH}")
else:
    print(f"ℹ️  Backup already exists: {BACKUP_PATH}")
    # Restore from backup before patching (so we always patch the original)
    import shutil
    shutil.copy2(BACKUP_PATH, MODEL_PATH)
    print("✅ Restored original from backup")

# ── Step 2: Read and deeply fix the model config ───────────────
with h5py.File(MODEL_PATH, 'r+') as f:
    raw = f.attrs.get('model_config')
    if isinstance(raw, bytes):
        raw = raw.decode('utf-8')

    # Fix 1: batch_shape / shape as string → proper key
    raw = re.sub(r'"batch_shape"\s*:', '"batch_input_shape":', raw)
    raw = re.sub(r'"shape"\s*:\s*(\[[^\]]*\])', r'"batch_input_shape": \1', raw)

    # Fix 2: Remove synchronized
    raw = re.sub(r',?\s*"synchronized"\s*:\s*(true|false)', '', raw)

    # Fix 3: Remove ragged
    raw = re.sub(r',?\s*"ragged"\s*:\s*(true|false)', '', raw)

    # Fix 4: Parse JSON and fix any batch_input_shape that is a string instead of list
    config = json.loads(raw)

    def deep_fix(obj):
        if isinstance(obj, dict):
            # Fix batch_input_shape if it's a string like "[null, 512, 512, 4]"
            for key in ['batch_input_shape', 'batch_shape', 'shape']:
                if key in obj and isinstance(obj[key], str):
                    try:
                        obj[key] = json.loads(obj[key])
                    except Exception:
                        pass
            # Remove bad keys
            for bad in ['synchronized', 'ragged']:
                obj.pop(bad, None)
            # Recurse
            for v in obj.values():
                deep_fix(v)
        elif isinstance(obj, list):
            for item in obj:
                deep_fix(item)

    deep_fix(config)

    fixed = json.dumps(config)
    f.attrs['model_config'] = fixed.encode('utf-8')

    # Also fix layer configs stored inside the file
    def fix_group(group):
        for key in group.keys():
            item = group[key]
            if isinstance(item, h5py.Group):
                if 'config' in item.attrs:
                    cfg_raw = item.attrs['config']
                    if isinstance(cfg_raw, bytes):
                        cfg_raw = cfg_raw.decode('utf-8')
                    cfg_raw = re.sub(r'"batch_shape"\s*:', '"batch_input_shape":', cfg_raw)
                    cfg_raw = re.sub(r',?\s*"synchronized"\s*:\s*(true|false)', '', cfg_raw)
                    cfg_raw = re.sub(r',?\s*"ragged"\s*:\s*(true|false)', '', cfg_raw)
                    item.attrs['config'] = cfg_raw.encode('utf-8')
                fix_group(item)

    fix_group(f)
    print("✅ Model fully patched and saved")

print("\nNow restart uvicorn — model should load successfully.")