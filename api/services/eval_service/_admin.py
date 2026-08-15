"""Admin gate for the eval routes.

Split out of the package ``__init__`` to keep it inside the 250-line
``api/services/**`` budget. Nothing here touches ``_eval_root`` /
``_eval_run_id``, the two seams test fixtures monkeypatch on the package
module, so it relocates without changing what a patch reaches.
"""

from __future__ import annotations

import os


def require_admin(request) -> None:
    """Raise HTTPException(403) unless the request bears a valid EVAL_ADMIN_TOKEN.

    Acceptable to raise HTTP-shaped exceptions from the service tier: the gate
    is tiny and reused by every eval route.
    """
    from fastapi import HTTPException, status

    expected = os.getenv("EVAL_ADMIN_TOKEN", "")
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Eval set builder is disabled. Set EVAL_ADMIN_TOKEN to enable.",
        )
    token = request.cookies.get("eval_admin") or request.query_params.get("token") or ""
    if token != expected:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Eval set builder requires a valid admin token.",
        )
