# Sample Package Manifest

**Generado**: 2026-02-17
**Propósito**: paquete versionable para reproducibilidad offline

## Dataset

| Campo | Valor |
|-------|-------|
| n_rounds | 10,000 |
| n_actions | 80 |
| context_dim | 20 (c0..c19) |
| seed | 42 |
| split_strategy | time |
| mapping_type | identity |
| is_identity_map | true |

## Splits

| Split | Muestras |
|-------|----------|
| train | 6,300 |
| val | 700 |
| test | 3,000 |

**split_manifest_hash**: `07db61136a04a92b6643c86d811df4ea71304770c6410cb997425e5a8591cf88a`

## Archivos en data/sample/

| Archivo | Origen | Contenido |
|---------|--------|-----------|
| bandit_feedback_sample.npz | bandit_feedback/bandit_feedback.npz | Arrays OBP (context, action, reward, position, pscore) |
| metadata_sample.json | bandit_feedback/metadata.json | Metadata completo con fingerprint |
| action_id_map_sample.json | bandit_feedback/action_id_map.json | Mapa de action IDs (identity) |
| split_manifest_sample.npz | splits/split_manifest.npz | Índices train/val/test (NPZ canónico) |
| split_manifest_sample.json | bandit_feedback/split_manifest.json | Índices train/val/test (JSON inspección) |
| obd_time_sample.inter | recbole_atomic/obd_time.inter | Interactions con timestamp |
| obd_sample.user | recbole_atomic/obd.user | User features c0..c19 |
| obd_sample.item | recbole_atomic/obd.item | Item features i0..i3 |

## Cómo regenerar

```bash
# Regenerar artefactos desde cero
make data
# o directamente:
python -m src.data.build_obd_datasets --mode sample

# Re-empaquetar sample (manual)
cp data/bandit_feedback/bandit_feedback.npz  data/sample/bandit_feedback_sample.npz
cp data/bandit_feedback/metadata.json        data/sample/metadata_sample.json
cp data/bandit_feedback/action_id_map.json   data/sample/action_id_map_sample.json
cp data/bandit_feedback/split_manifest.json  data/sample/split_manifest_sample.json
cp data/splits/split_manifest.npz            data/sample/split_manifest_sample.npz
cp data/recbole_atomic/obd_time.inter        data/sample/obd_time_sample.inter
cp data/recbole_atomic/obd.user              data/sample/obd_sample.user
cp data/recbole_atomic/obd.item              data/sample/obd_sample.item
```

## Política de versionamiento

- **SÍ commitear**: `data/sample/` (este directorio completo)
- **NO commitear**: `data/raw/`, `data/interim/`, `data/bandit_feedback/`, `data/recbole_atomic/`, `data/splits/`
- Controlado por `.gitignore` con excepción explícita `!data/sample/**`
