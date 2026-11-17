import shlex
import subprocess
from pathlib import Path
from langchain_core.tools import tool

# Kořenový adresář repozitáře (o úroveň výš než orchestrator/)
REPO_ROOT = Path(__file__).resolve().parent.parent

# Povolené cíle pro zápis souborů
ALLOWED_WRITE_PREFIXES = (
    "backend/",
    "frontend/",
    "ARCHITECTURE.md",
    "README.md",
)

# Povolené příkazy pro QA agenta
ALLOWED_QA_COMMAND_PREFIXES = (
    "pytest",
    "python -m pytest",
    "ruff",
    "npm run build",
    "npm run lint",
    "npm test",
    "npx tsc",
)

# Developer může spustit pouze kompilaci Python souborů v backendu.
ALLOWED_DEVELOPER_COMMAND_PREFIXES = (
    "python -m compileall -q .",
    "npm run build",
)


def _resolve_safe_path(relative_path: str) -> Path:
    """Převede relativní cestu na absolutní a ověří, že neopouští REPO_ROOT."""
    clean_rel = relative_path.lstrip("/\\")
    target = (REPO_ROOT / clean_rel).resolve()
    if not str(target).startswith(str(REPO_ROOT)):
        raise ValueError(f"Přístup mimo kořen repozitáře je zakázán: {relative_path}")
    return target


def _is_write_allowed(relative_path: str) -> bool:
    """Kontroluje, zda agent smí zapisovat do dané cesty."""
    normalized = relative_path.replace("\\", "/").lstrip("/")
    if any(part in normalized.split("/") for part in (".git", ".env", "orchestrator")):
        return False
    return normalized.startswith(ALLOWED_WRITE_PREFIXES)


@tool
def read_file(file_path: str) -> str:
    """Přečte obsah souboru z repozitáře. Zadej relativní cestu od kořene (např. 'backend/main.py')."""
    try:
        target = _resolve_safe_path(file_path)
        if not target.exists() or not target.is_file():
            return f"Chyba: Soubor '{file_path}' neexistuje."
        return target.read_text(encoding="utf-8")
    except Exception as e:
        return f"Chyba při čtení souboru '{file_path}': {e}"


@tool
def write_file(file_path: str, content: str) -> str:
    """Zapíše textový obsah do souboru v repozitáři. Vytvoří i neexistující nadřazené složky."""
    try:
        if not _is_write_allowed(file_path):
            return (
                f"Chyba oprávnění: Zápis do '{file_path}' není povolen. "
                f"Zapisovat lze pouze do: {', '.join(ALLOWED_WRITE_PREFIXES)}"
            )
        target = _resolve_safe_path(file_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"OK: Soubor '{file_path}' byl úspěšně uložen ({len(content)} znaků)."
    except Exception as e:
        return f"Chyba při zápisu souboru '{file_path}': {e}"


@tool
def list_directory(dir_path: str = ".") -> str:
    """Vypíše seznam souborů a podsložek v zadaném adresáři repozitáře."""
    try:
        target = _resolve_safe_path(dir_path)
        if not target.exists() or not target.is_dir():
            return f"Chyba: Adresář '{dir_path}' neexistuje."

        entries = []
        for item in sorted(target.rglob("*")):
            rel = item.relative_to(REPO_ROOT).as_posix()
            if any(ignored in rel.split("/") for ignored in (".git", ".venv", "node_modules", "__pycache__")):
                continue
            prefix = "[DIR] " if item.is_dir() else "[FILE]"
            entries.append(f"{prefix} {rel}")

        return "\n".join(entries) if entries else "Adresář je prázdný."
    except Exception as e:
        return f"Chyba při výpisu adresáře '{dir_path}': {e}"


def _run_terminal_command(
    command: str,
    working_dir: str,
    allowed_command_prefixes: tuple[str, ...],
    allowed_working_dirs: tuple[str, ...],
    *,
    exact_command: bool = False,
) -> str:
    if working_dir not in allowed_working_dirs:
        return (
            f"Chyba: working_dir musí být jedna z hodnot "
            f"{', '.join(repr(path) for path in allowed_working_dirs)}."
        )

    cmd_stripped = command.strip()
    try:
        command_args = shlex.split(cmd_stripped)
    except ValueError as e:
        return f"Chyba syntaxe příkazu '{cmd_stripped}': {e}"

    allowed_prefix_args = tuple(
        tuple(shlex.split(prefix)) for prefix in allowed_command_prefixes
    )
    command_is_allowed = any(
        tuple(command_args[: len(prefix_args)]) == prefix_args
        and (not exact_command or len(command_args) == len(prefix_args))
        for prefix_args in allowed_prefix_args
    )
    if not command_is_allowed:
        return (
            f"Chyba oprávnění: Příkaz '{cmd_stripped}' není na seznamu povolených příkazů "
            f"({', '.join(allowed_command_prefixes)})."
        )

    cwd = _resolve_safe_path(working_dir)
    if not cwd.is_dir():
        return (
            f"Chyba: Pracovní adresář '{working_dir}' neexistuje "
            "nebo není adresářem."
        )

    try:
        result = subprocess.run(
            command_args,
            shell=False,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=120,
        )
        output = f"Exit code: {result.returncode}\n"
        if result.stdout:
            output += f"STDOUT:\n{result.stdout}\n"
        if result.stderr:
            output += f"STDERR:\n{result.stderr}\n"
        # Oříznutí příliš dlouhých logů kvůli limitu kontextového okna
        return output[:8000]
    except subprocess.TimeoutExpired:
        return f"Chyba: Příkaz '{cmd_stripped}' vypršel po 120 sekundách."
    except Exception as e:
        return f"Chyba při spouštění příkazu: {e}"


@tool
def run_terminal_command(command: str, working_dir: str) -> str:
    """Spustí povolený QA příkaz ve složce 'backend' nebo 'frontend'."""
    return _run_terminal_command(
        command,
        working_dir,
        ALLOWED_QA_COMMAND_PREFIXES,
        ("backend", "frontend"),
    )


@tool
def run_developer_terminal_command(command: str, working_dir: str) -> str:
    """Zkompiluje Python soubory v backendu příkazem 'python -m compileall -q .'."""
    return _run_terminal_command(
        command,
        working_dir,
        ALLOWED_DEVELOPER_COMMAND_PREFIXES,
        ("backend", "frontend"),
        exact_command=True,
    )


# Sady nástrojů přiřazené jednotlivým rolím agentů
ARCHITECT_TOOLS = [read_file, write_file, list_directory]
DEVELOPER_TOOLS = [
    read_file,
    write_file,
    list_directory,
    run_developer_terminal_command,
]
QA_TOOLS = [read_file, list_directory, run_terminal_command]
