"""
Admin CLI for the versioned indices behind the `document-chunks` alias.

    python index_admin.py status              # alias -> which version, every version + doc count
    python index_admin.py migrate             # ONE-TIME: legacy concrete index -> document-chunks-v1 + alias
    python index_admin.py reindex             # new version with the current INDEX_MAPPING, copy data, swap alias
    python index_admin.py reindex --delete-old
    python index_admin.py switch document-chunks-v1   # point the alias at another version (rollback)
    python index_admin.py delete document-chunks-v1   # drop an old version the alias no longer uses

Typical mapping change (e.g. a different analyzer for `text`):
    1. edit INDEX_MAPPING in es_index.py
    2. python index_admin.py reindex      -> builds v(N+1), copies every doc, swaps the alias
    3. check the app; if something is off: python index_admin.py switch <old version>
    4. once happy: python index_admin.py delete <old version>

Reindex copies `_source`, which still holds both embeddings, so nothing is re-embedded.
That covers analyzer/field-type/settings changes. Switching to a different embedding
model (new vectors) is NOT a reindex -- that needs re-embedding every chunk.

Don't upload documents while migrate/reindex is running: writes land in the old index
and won't be in the copy (the doc-count check after copying would catch it and abort
before the alias swap).
"""

import argparse
import re
import sys

from elasticsearch import Elasticsearch

from es_index import ES_HOST, INDEX_MAPPING, INDEX_NAME, versioned_index

VERSION_RE = re.compile(rf"^{re.escape(INDEX_NAME)}-v(\d+)$")
REINDEX_TIMEOUT_S = 3600


def alias_targets(es) -> list[str]:
    """Concrete indices the alias currently points at ([] if the alias doesn't exist)."""
    if not es.indices.exists_alias(name=INDEX_NAME):
        return []
    return sorted(es.indices.get_alias(name=INDEX_NAME).keys())


def is_legacy_concrete_index(es) -> bool:
    """True if INDEX_NAME is still a plain index (pre-alias layout)."""
    return es.indices.exists(index=INDEX_NAME) and not es.indices.exists_alias(name=INDEX_NAME)


def list_versions(es) -> list[dict]:
    """Every document-chunks-vN index, oldest first: [{index, version, docs, size}, ...]."""
    rows = es.cat.indices(index=f"{INDEX_NAME}-v*", format="json", h="index,docs.count,store.size")
    versions = []
    for row in rows:
        m = VERSION_RE.match(row["index"])
        if m:
            versions.append({
                "index": row["index"],
                "version": int(m.group(1)),
                "docs": int(row["docs.count"] or 0),
                "size": row["store.size"],
            })
    return sorted(versions, key=lambda v: v["version"])


def next_version_name(es) -> str:
    versions = list_versions(es)
    return versioned_index(versions[-1]["version"] + 1 if versions else 1)


def count(es, index) -> int:
    es.indices.refresh(index=index)
    return es.count(index=index)["count"]


def create_version(es, index):
    es.indices.create(index=index, settings=INDEX_MAPPING["settings"], mappings=INDEX_MAPPING["mappings"])
    print(f"  created {index} with the current INDEX_MAPPING")


def copy_documents(es, source, dest):
    """_reindex source -> dest, then verify nothing was lost. Raises before any alias
    change if the copy is incomplete, so the live alias is never pointed at bad data."""
    print(f"  copying {source} -> {dest} ...")
    resp = es.options(request_timeout=REINDEX_TIMEOUT_S).reindex(
        source={"index": source},
        dest={"index": dest},
        wait_for_completion=True,
        refresh=True,
    )
    if resp.get("failures"):
        raise RuntimeError(f"reindex reported {len(resp['failures'])} failures, first: {resp['failures'][0]}")

    src_count, dest_count = count(es, source), count(es, dest)
    print(f"  copied {resp['created'] + resp['updated']} docs ({source}: {src_count}, {dest}: {dest_count})")
    if src_count != dest_count:
        raise RuntimeError(
            f"doc count mismatch ({source}: {src_count} vs {dest}: {dest_count}) -- was something "
            f"indexed during the copy? Alias left unchanged; {dest} can be deleted and the command re-run."
        )


def confirm(question, assume_yes) -> bool:
    if assume_yes:
        return True
    return input(f"{question} Type 'yes' to continue: ").strip().lower() == "yes"


def cmd_status(es, args):
    targets = alias_targets(es)
    if targets:
        print(f"alias '{INDEX_NAME}' -> {', '.join(targets)}")
    elif is_legacy_concrete_index(es):
        print(f"'{INDEX_NAME}' is a LEGACY concrete index ({count(es, INDEX_NAME)} docs), not an alias.")
        print("  run: python index_admin.py migrate")
    else:
        print(f"'{INDEX_NAME}' doesn't exist yet (the app creates {versioned_index(1)} + alias on first upload).")

    versions = list_versions(es)
    if versions:
        print("\nversions:")
        for v in versions:
            marker = "  <- live" if v["index"] in targets else ""
            print(f"  {v['index']:<28} {v['docs']:>8} docs  {v['size']:>8}{marker}")


def cmd_migrate(es, args):
    if es.indices.exists_alias(name=INDEX_NAME):
        print(f"Already migrated: alias '{INDEX_NAME}' -> {', '.join(alias_targets(es))}")
        return
    if not es.indices.exists(index=INDEX_NAME):
        print(f"No '{INDEX_NAME}' index to migrate -- the app will create {versioned_index(1)} + alias itself.")
        return

    dest = next_version_name(es)
    print(f"Plan: copy legacy index '{INDEX_NAME}' into '{dest}', then in ONE atomic step delete")
    print(f"      '{INDEX_NAME}' and create alias '{INDEX_NAME}' -> '{dest}'.")
    print("      (An alias can't share its name with an index, so the legacy index has to go;")
    print("       the data is safe in the verified copy.)")
    if not confirm("Proceed?", args.yes):
        print("Aborted.")
        return

    create_version(es, dest)
    copy_documents(es, INDEX_NAME, dest)
    es.indices.update_aliases(actions=[
        {"remove_index": {"index": INDEX_NAME}},
        {"add": {"index": dest, "alias": INDEX_NAME, "is_write_index": True}},
    ])
    print(f"Done: alias '{INDEX_NAME}' -> '{dest}'.")


def cmd_reindex(es, args):
    targets = alias_targets(es)
    if not targets:
        sys.exit(f"'{INDEX_NAME}' is not an alias yet -- run `python index_admin.py migrate` first.")
    if len(targets) > 1:
        sys.exit(f"alias points at several indices ({targets}) -- fix with `switch` first.")
    current = targets[0]

    dest = next_version_name(es)
    print(f"Plan: build '{dest}' with the current INDEX_MAPPING, copy every doc from '{current}',")
    print(f"      then swap alias '{INDEX_NAME}' -> '{dest}' atomically.")
    print(f"      '{current}' is {'DELETED afterwards' if args.delete_old else 'kept for rollback'}.")
    if not confirm("Proceed?", args.yes):
        print("Aborted.")
        return

    create_version(es, dest)
    copy_documents(es, current, dest)
    es.indices.update_aliases(actions=[
        {"remove": {"index": current, "alias": INDEX_NAME}},
        {"add": {"index": dest, "alias": INDEX_NAME, "is_write_index": True}},
    ])
    print(f"Done: alias '{INDEX_NAME}' -> '{dest}'.")

    if args.delete_old:
        es.indices.delete(index=current)
        print(f"Deleted '{current}'.")
    else:
        print(f"Rollback if needed: python index_admin.py switch {current}")


def cmd_switch(es, args):
    target = args.index
    if not VERSION_RE.match(target) or not es.indices.exists(index=target):
        sys.exit(f"'{target}' is not an existing {INDEX_NAME}-vN index (see `status`).")

    current = alias_targets(es)
    if current == [target]:
        print(f"Alias already points at '{target}'.")
        return
    actions = [{"remove": {"index": idx, "alias": INDEX_NAME}} for idx in current]
    actions.append({"add": {"index": target, "alias": INDEX_NAME, "is_write_index": True}})
    es.indices.update_aliases(actions=actions)
    print(f"Alias '{INDEX_NAME}': {', '.join(current) or '(none)'} -> '{target}'.")


def cmd_delete(es, args):
    target = args.index
    if not VERSION_RE.match(target) or not es.indices.exists(index=target):
        sys.exit(f"'{target}' is not an existing {INDEX_NAME}-vN index (see `status`).")
    if target in alias_targets(es):
        sys.exit(f"'{target}' is the LIVE index behind the alias -- `switch` to another version first.")
    if not confirm(f"Permanently delete '{target}' ({count(es, target)} docs)?", args.yes):
        print("Aborted.")
        return
    es.indices.delete(index=target)
    print(f"Deleted '{target}'.")


def main():
    parser = argparse.ArgumentParser(description=f"Manage the versioned indices behind the '{INDEX_NAME}' alias.")
    parser.add_argument("--host", default=ES_HOST)
    parser.add_argument("-y", "--yes", action="store_true", help="skip the confirmation prompt")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="show the alias and every version").set_defaults(func=cmd_status)
    sub.add_parser("migrate", help="one-time: legacy concrete index -> v1 + alias").set_defaults(func=cmd_migrate)
    p = sub.add_parser("reindex", help="build the next version with the current mapping and swap the alias")
    p.add_argument("--delete-old", action="store_true", help="delete the previous version after the swap")
    p.set_defaults(func=cmd_reindex)
    p = sub.add_parser("switch", help="point the alias at another existing version (rollback)")
    p.add_argument("index")
    p.set_defaults(func=cmd_switch)
    p = sub.add_parser("delete", help="delete a version the alias no longer points at")
    p.add_argument("index")
    p.set_defaults(func=cmd_delete)

    args = parser.parse_args()
    es = Elasticsearch(args.host)
    args.func(es, args)


if __name__ == "__main__":
    main()
