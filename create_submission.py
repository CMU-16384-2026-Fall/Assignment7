import os
import zipfile

# Kept at their folder paths in the zip, as in the handout.
CODE = [
    os.path.join("ex1", "robot.py"),
]


def create_submission():
    missing = [path for path in CODE if not os.path.exists(path)]
    if missing:
        raise SystemExit(
            "Missing " + ", ".join(missing) + " — run this from the assignment root."
        )

    with zipfile.ZipFile("handin.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        for path in CODE:
            zf.write(path, path)

    print("Created submission archive: handin.zip")
    for path in CODE:
        print(f"  {path}")


if __name__ == "__main__":
    create_submission()
