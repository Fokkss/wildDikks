# Repository cleanup checklist

Recommended final structure:

```text
README.md
requirements.txt
Dockerfile
pytest.ini
data/                         # local only, do not commit private data
wind_beta_0_11/
  README.md
  WORK_LOG.md
  FOR_VAAS.md
  new_model_beta13/
  scripts_beta13/
  configs_beta13/
```

Do not commit:

```text
artifacts_beta*/
reports_beta*/
submissions_beta*/
*.pkl
*.joblib
*.csv submission outputs, unless required by the platform
private train/valid data
```

Add to `.gitignore`:

```gitignore
data/
artifacts*/
reports*/
submissions*/
*.pkl
*.joblib
*.csv
!requirements.txt
```

Root `README.md` should stop being a placeholder and should explain:

```text
1. task statement;
2. model idea;
3. physics features;
4. train command;
5. predict command;
6. where output submission is saved;
7. current best reproducible score and caveat about calibrated diagnostic artifacts.
```
