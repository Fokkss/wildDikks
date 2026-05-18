from __future__ import annotations

import argparse
import shutil
from pathlib import Path


DEFAULT_CODE_PATHS = [
    "new_model_v1_0",
    "configs_v1_0",
    "configs_v1_6",
    "scripts",
    "requirements.txt",
    "requirements_final_v1_6.txt",
]


def _copy_any(src: Path, dst: Path) -> None:
    # копируем файл или папку в сборочную директорию
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    elif src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="make final delivery zip")
    parser.add_argument("--project_model_dir", default="model", help="path to model folder")
    parser.add_argument("--prediction_csv", required=True, help="final 18.05 forecast csv")
    parser.add_argument("--artifact_dir", required=True, help="trained model artifact directory")
    parser.add_argument("--train_data", required=True, help="used train csv")
    parser.add_argument("--note_docx", required=True, help="explanatory note docx")
    parser.add_argument("--output_zip", default="final_delivery.zip")
    args = parser.parse_args()

    model_dir = Path(args.project_model_dir)
    build_dir = Path("_final_delivery_build")

    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True)

    # прогноз на 18.05 кладем в корень архива
    _copy_any(Path(args.prediction_csv), build_dir / Path(args.prediction_csv).name)

    # веса и json-карточки нужны только в zip, а не в публичном github
    _copy_any(Path(args.artifact_dir), build_dir / "artifacts")

    # по тз использованные данные обучения должны лежать в zip
    _copy_any(Path(args.train_data), build_dir / "data" / Path(args.train_data).name)

    # пояснительная записка добавляется как отдельный файл
    _copy_any(Path(args.note_docx), build_dir / Path(args.note_docx).name)

    # код копируется отдельно, чтобы архив был воспроизводимым
    code_root = build_dir / "code" / "model"
    for rel in DEFAULT_CODE_PATHS:
        src = model_dir / rel
        if src.exists():
            _copy_any(src, code_root / rel)

    output_base = Path(args.output_zip)
    if output_base.suffix == ".zip":
        output_base = output_base.with_suffix("")

    archive_path = shutil.make_archive(str(output_base), "zip", build_dir)
    shutil.rmtree(build_dir)

    print(f"saved delivery zip: {archive_path}")


if __name__ == "__main__":
    main()
