"""Collect full Nsight Compute reports for one baseline and candidate launch."""

import base64
import io
import json
import subprocess
import zipfile


def run(mode, kernel_pattern):
    report = f"/root/{mode}"
    command = [
        "ncu",
        "--set",
        "full",
        "--target-processes",
        "all",
        "--kernel-name",
        f"regex:{kernel_pattern}",
        "--launch-count",
        "1",
        "--force-overwrite",
        "--export",
        report,
        "python",
        "-u",
        "/root/ncu_workload.py",
        mode,
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode:
        raise RuntimeError(
            f"ncu {mode} failed ({completed.returncode})\n"
            + completed.stdout
            + completed.stderr
        )
    imported = subprocess.run(
        ["ncu", "--import", report + ".ncu-rep", "--csv", "--page", "raw"],
        text=True,
        capture_output=True,
        check=True,
    )
    return {
        "mode": mode,
        "command": command,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "csv": imported.stdout,
        "report": report + ".ncu-rep",
    }


def main():
    rows = [
        run("baseline", "_panel_inner32$"),
        run("candidate", "_panel_inner32_subtile64$"),
    ]
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for row in rows:
            mode = row["mode"]
            archive.write(row["report"], f"{mode}.ncu-rep")
            archive.writestr(f"{mode}.csv", row["csv"])
            archive.writestr(f"{mode}.stdout.txt", row["stdout"])
            archive.writestr(f"{mode}.stderr.txt", row["stderr"])
    payload = base64.b64encode(stream.getvalue()).decode()
    print(
        "NCU_META:"
        + json.dumps(
            {
                row["mode"]: {
                    "command": row["command"],
                    "stdout": row["stdout"],
                    "stderr": row["stderr"],
                }
                for row in rows
            }
        ),
        flush=True,
    )
    for index, offset in enumerate(range(0, len(payload), 8192)):
        print(f"NCU_B64_PART:{index}:" + payload[offset : offset + 8192], flush=True)


if __name__ == "__main__":
    main()
