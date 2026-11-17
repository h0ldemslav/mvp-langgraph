from typing import Literal

from pydantic import BaseModel, Field


class QADecision(BaseModel):
    """Strukturovaný výstup QA agenta pro řízení zpětnovazební smyčky."""

    passed: bool = Field(
        description="True, pokud backend testy i frontend build prošly bez chyb."
    )
    retry_target: Literal["backend_dev", "frontend_dev", "none"] = Field(
        description="Kam vrátit úkol k opravě. Pokud passed=True, nastav 'none'."
    )
    summary: str = Field(
        description="Stručný popis nalezených chyb nebo potvrzení, že vše prošlo."
    )
