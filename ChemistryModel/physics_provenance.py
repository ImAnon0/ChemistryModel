"""Reproducible source identity for a concrete ChemistryModel implementation.

The manifest is derived from the selected class rather than a maintained list.
It includes class/MRO source files, files that supplied copied method objects,
and the transitive local-import closure of those files.  That matters for the
unified-radial reference, whose effective implementation includes methods
defined in research modules as well as constants imported from helper modules.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def _inside_project(path: Path, root: Path) -> bool:
    try:
        relative = path.resolve().relative_to(root)
    except ValueError:
        return False
    excluded = {".git", ".venv", "site-packages", "__pycache__"}
    return not any(part in excluded for part in relative.parts)


def _source_path(value, root: Path) -> Path | None:
    try:
        source = inspect.getsourcefile(value)
    except (TypeError, OSError):
        return None
    if not source:
        return None
    path = Path(source).resolve()
    return path if path.is_file() and _inside_project(path, root) else None


def _descriptor_callables(value):
    if isinstance(value, (staticmethod, classmethod)):
        return (value.__func__,)
    if isinstance(value, property):
        return tuple(
            item for item in (value.fget, value.fset, value.fdel)
            if item is not None
        )
    return (value,) if inspect.isfunction(value) else ()


def _callable_source_files(simulation_class, root: Path) -> set[Path]:
    """Find class and copied-method definitions that determine behaviour."""

    files: set[Path] = set()
    queue = []
    seen = set()
    for cls in simulation_class.__mro__:
        path = _source_path(cls, root)
        if path is not None:
            files.add(path)
        for value in cls.__dict__.values():
            queue.extend(_descriptor_callables(value))

    while queue:
        function = queue.pop()
        marker = id(function)
        if marker in seen:
            continue
        seen.add(marker)
        path = _source_path(function, root)
        if path is not None:
            files.add(path)

        code = getattr(function, "__code__", None)
        namespace = getattr(function, "__globals__", {})
        if code is None:
            continue
        for name in code.co_names:
            referenced = namespace.get(name)
            if inspect.isfunction(referenced):
                referenced_path = _source_path(referenced, root)
                if referenced_path is not None:
                    queue.append(referenced)
            elif inspect.isclass(referenced):
                referenced_path = _source_path(referenced, root)
                if referenced_path is not None:
                    files.add(referenced_path)
                    for value in referenced.__dict__.values():
                        queue.extend(_descriptor_callables(value))
            elif inspect.ismodule(referenced):
                referenced_path = _source_path(referenced, root)
                if referenced_path is not None:
                    files.add(referenced_path)
    return files


def _module_name(path: Path, root: Path) -> tuple[str, bool]:
    relative = path.relative_to(root)
    parts = list(relative.with_suffix("").parts)
    is_package = bool(parts and parts[-1] == "__init__")
    if is_package:
        parts.pop()
    return ".".join(parts), is_package


def _module_path(name: str, root: Path) -> Path | None:
    if not name:
        return None
    plain = root.joinpath(*name.split(".")).with_suffix(".py")
    if plain.is_file():
        return plain.resolve()
    package = root.joinpath(*name.split("."), "__init__.py")
    if package.is_file():
        return package.resolve()
    return None


def _local_imports(path: Path, root: Path) -> set[Path]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return set()
    current_module, is_package = _module_name(path, root)
    package_parts = current_module.split(".") if is_package else current_module.split(".")[:-1]
    result: set[Path] = set()
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                remove = max(node.level - 1, 0)
                base = package_parts[:len(package_parts) - remove]
                module_parts = node.module.split(".") if node.module else []
                name = ".".join((*base, *module_parts))
            else:
                name = node.module or ""
            if name:
                names.append(name)
        for name in names:
            imported = _module_path(name, root)
            if imported is not None:
                result.add(imported)
    return result


def effective_source_manifest(simulation_class, project_root=None) -> list[str]:
    """Return sorted project-relative sources affecting ``simulation_class``."""

    root = Path(project_root or PROJECT_ROOT).resolve()
    files = _callable_source_files(simulation_class, root)
    queue = list(files)
    while queue:
        path = queue.pop()
        for imported in _local_imports(path, root):
            if imported not in files:
                files.add(imported)
                queue.append(imported)
    return sorted(path.relative_to(root).as_posix() for path in files)


def physics_source_identity(simulation_class, project_root=None) -> dict:
    """Return a content hash and auditable manifest for selected physics."""

    root = Path(project_root or PROJECT_ROOT).resolve()
    files = effective_source_manifest(simulation_class, root)
    digest = hashlib.sha256()
    for relative in files:
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update((root / relative).read_bytes())
        digest.update(b"\0")
    return {
        "algorithm": "chemistrymodel-effective-sources-v1",
        "sha256": digest.hexdigest(),
        "files": files,
    }
