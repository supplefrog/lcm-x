"""Restore a pinned candidate on the runner without publishing its Git objects."""

import base64
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile


def git(*arguments):
    return subprocess.check_output(["git", "-C", "follow-next", *arguments], text=True).strip()


def oid(value):
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError("Expected a full Git object ID")
    return value


def restore(chain, published):
    if oid(chain["published"]) != published:
        raise ValueError("Bundle chain is based on another PR revision")
    final = oid(chain["final"])
    for upstream in chain["upstream"]:
        git("fetch", "--quiet", "--no-tags", "--depth=1", "https://github.com/electricsheephq/lcm-x.git", oid(upstream))

    with tempfile.TemporaryDirectory(dir=os.environ.get("RUNNER_TEMP")) as scratch:
        for i, item in enumerate(chain["bundles"]):
            expected = oid(item["commit"])
            data = base64.b64decode(item["data"], validate=True)
            if hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise ValueError("Bundle checksum mismatch")
            bundle = Path(scratch, f"candidate-{i}.bundle")
            bundle.write_bytes(data)
            commit, ref = git("bundle", "list-heads", str(bundle)).split()
            if commit != expected or not (ref == "HEAD" or ref.startswith("refs/heads/")):
                raise ValueError("Bundle does not advertise the expected commit")
            if ref != "HEAD":
                git("check-ref-format", ref)
            destination = f"refs/heads/verification-{i}"
            subprocess.run(["git", "-C", "follow-next", "bundle", "verify", str(bundle)], check=True, capture_output=True)
            git("fetch", "--quiet", str(bundle), f"{ref}:{destination}")
            if git("rev-parse", destination) != expected:
                raise ValueError("Restored commit does not match the bundle manifest")
    git("merge-base", "--is-ancestor", published, final)
    git("checkout", "--quiet", "--detach", final)
    if git("rev-parse", "HEAD") != final:
        raise ValueError("Did not check out the exact final commit")
    return final


def main():
    published = oid(os.environ["PR_HEAD_SHA"])
    if git("rev-parse", "HEAD") != published:
        raise SystemExit("The checkout does not match the pinned public PR head")
    secret = os.environ.pop("CANDIDATE_CHAIN", "")
    tested = restore(json.loads(secret), published) if secret else published
    with Path(os.environ["GITHUB_ENV"]).open("a", encoding="utf-8") as target:
        target.write(f"TESTED_SHA={tested}\n")


if __name__ == "__main__":
    main()
