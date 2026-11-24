"""Trello Work-Order Dispatch Client for Singapore Town Council Tickets.

Creates and updates Trello cards for estate fault reports:
- Routine / Medium tickets go to the board's "To Do" list.
- High / Critical severity tickets go straight to the "Urgent" list with an SLA due date.
- When duplicate reports merge onto an existing ticket, posts a subscriber count comment.
- When staff update ticket status (/progress, /resolve, or dashboard buttons), posts a
  status comment and optionally moves the card to "In Progress" or "Done" if those lists exist.

Gracefully disabled when TRELLO_BOARD_ID is not configured in `.env`.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

logger = logging.getLogger("council-bot.trello")

API = "https://api.trello.com/1"
TIMEOUT_SECONDS = 15


def trello_date(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


@dataclass
class TrelloCardResult:
    card_id: str
    card_url: str
    list_name: str
    due_iso: str | None = None


class TrelloWorkOrders:
    """Async + sync Trello client for submitting and syncing Town Council work orders."""

    def __init__(
        self,
        key: str,
        token: str,
        todo_list: str,
        urgent_list: str,
        in_progress_list: str | None = None,
        done_list: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        sync_transport: httpx.BaseTransport | None = None,
        *,
        critical_hours: float = 1.0,
        high_hours: float = 4.0,
        medium_hours: float = 24.0,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        # OAuth header keeps credentials out of query strings and server logs
        self.headers = {
            "Authorization": f'OAuth oauth_consumer_key="{key}", oauth_token="{token}"'
        }
        self.todo_list = todo_list
        self.urgent_list = urgent_list
        self.in_progress_list = in_progress_list
        self.done_list = done_list
        self.transport = transport
        self.sync_transport = sync_transport
        self.due_hours = {
            "critical": critical_hours,
            "emergency": critical_hours,
            "high": high_hours,
            "medium": medium_hours,
        }
        self.now = now

    @classmethod
    def from_env(
        cls,
        transport: httpx.BaseTransport | None = None,
        async_transport: httpx.AsyncBaseTransport | None = None,
    ) -> "TrelloWorkOrders | None":
        """Resolves list IDs from the configured Trello board. Returns None if disabled."""
        board = os.getenv("TRELLO_BOARD_ID", "").strip()
        if not board:
            return None

        key = os.getenv("TRELLO_API_KEY", "").strip()
        token = os.getenv("TRELLO_TOKEN", "").strip()
        if not key or not token:
            raise ValueError("Set TRELLO_API_KEY and TRELLO_TOKEN when TRELLO_BOARD_ID is set.")

        required_lists = {
            "TRELLO_TODO_LIST": os.getenv("TRELLO_TODO_LIST", "To Do").strip() or "To Do",
            "TRELLO_URGENT_LIST": os.getenv("TRELLO_URGENT_LIST", "Urgent").strip() or "Urgent",
        }
        optional_lists = {
            "TRELLO_IN_PROGRESS_LIST": os.getenv("TRELLO_IN_PROGRESS_LIST", "In Progress").strip() or "In Progress",
            "TRELLO_DONE_LIST": os.getenv("TRELLO_DONE_LIST", "Done").strip() or "Done",
        }

        hours: dict[str, float] = {}
        for env_key, default_val in (
            ("TRELLO_DUE_CRITICAL_HOURS", "1"),
            ("TRELLO_DUE_HIGH_HOURS", "4"),
            ("TRELLO_DUE_MEDIUM_HOURS", "24"),
        ):
            raw = os.getenv(env_key, default_val).strip() or default_val
            try:
                val = float(raw)
            except ValueError as exc:
                raise ValueError(f"{env_key} must be a non-negative number of hours.") from exc
            if val < 0:
                raise ValueError(f"{env_key} must be >= 0 (0 disables due dates).")
            hours[env_key] = val

        headers = {"Authorization": f'OAuth oauth_consumer_key="{key}", oauth_token="{token}"'}
        try:
            with httpx.Client(headers=headers, timeout=TIMEOUT_SECONDS, transport=transport) as client:
                response = client.get(f"{API}/boards/{board}/lists", params={"fields": "name"})
        except httpx.HTTPError as exc:
            raise ValueError(f"Could not reach Trello API ({type(exc).__name__}).") from exc

        if response.status_code in (400, 401, 403):
            raise ValueError("Trello rejected TRELLO_API_KEY, TRELLO_TOKEN, or board access.")
        if response.status_code == 404:
            raise ValueError(f"Trello board '{board}' was not found. Check TRELLO_BOARD_ID.")
        response.raise_for_status()

        board_lists = {
            item["name"].strip().casefold(): item["id"]
            for item in response.json()
            if isinstance(item, dict) and "name" in item and "id" in item
        }

        resolved_req: dict[str, str] = {}
        for env_key, list_name in required_lists.items():
            lid = board_lists.get(list_name.casefold())
            if not lid:
                raise ValueError(f'Trello board has no "{list_name}" list. Check {env_key}.')
            resolved_req[env_key] = lid

        in_prog_id = board_lists.get(optional_lists["TRELLO_IN_PROGRESS_LIST"].casefold())
        done_id = board_lists.get(optional_lists["TRELLO_DONE_LIST"].casefold())

        return cls(
            key=key,
            token=token,
            todo_list=resolved_req["TRELLO_TODO_LIST"],
            urgent_list=resolved_req["TRELLO_URGENT_LIST"],
            in_progress_list=in_prog_id,
            done_list=done_id,
            transport=async_transport,
            sync_transport=transport,
            critical_hours=hours["TRELLO_DUE_CRITICAL_HOURS"],
            high_hours=hours["TRELLO_DUE_HIGH_HOURS"],
            medium_hours=hours["TRELLO_DUE_MEDIUM_HOURS"],
        )

    def _compute_due(self, severity: str) -> str | None:
        hrs = self.due_hours.get((severity or "").strip().lower(), 0.0)
        if hrs <= 0:
            return None
        return trello_date(self.now() + timedelta(hours=hrs))

    @staticmethod
    def format_card_payload(
        ticket_id: int,
        category: str,
        description: str,
        location: str | None,
        town_council: str | None,
        severity: str = "normal",
        ai_diagnosis: str | None = None,
        reporter: str | None = None,
        lat: float | None = None,
        lng: float | None = None,
    ) -> tuple[str, str, bool]:
        sev_clean = (severity or "normal").strip()
        is_urgent = sev_clean.lower() in {"high", "critical", "emergency"}
        council_label = town_council or "Singapore Town Council"
        short_desc = " ".join(description.split())[:65]
        sev_tag = f"[{sev_clean.upper()}] " if is_urgent else ""
        title = f"#{ticket_id} {sev_tag}{category.title()} · {council_label} — {short_desc}"

        lines = [
            f"**Ticket ID:** #{ticket_id}",
            f"**Town Council:** {council_label}",
            f"**Category:** {category}",
            f"**Severity:** {sev_clean.upper()}",
            f"**Location:** {location or 'Not provided'}",
        ]
        if lat is not None and lng is not None:
            lines.append(f"**Map Coordinates:** {lat:.5f}, {lng:.5f} (https://maps.google.com/?q={lat:.5f},{lng:.5f})")
        if reporter:
            lines.append(f"**Reported By:** {reporter}")
        if ai_diagnosis:
            lines.append(f"**AI Vision Assessment:** {ai_diagnosis}")
        lines.extend(["", "**Resident Description:**", description])
        return title, "\n".join(lines), is_urgent

    async def submit_work_order(
        self,
        ticket_id: int,
        category: str,
        description: str,
        location: str | None,
        town_council: str | None,
        severity: str = "normal",
        ai_diagnosis: str | None = None,
        reporter: str | None = None,
        lat: float | None = None,
        lng: float | None = None,
    ) -> TrelloCardResult:
        """Creates a new Trello work-order card for a resident fault report."""
        title, desc, is_urgent = self.format_card_payload(
            ticket_id=ticket_id,
            category=category,
            description=description,
            location=location,
            town_council=town_council,
            severity=severity,
            ai_diagnosis=ai_diagnosis,
            reporter=reporter,
            lat=lat,
            lng=lng,
        )
        list_id = self.urgent_list if is_urgent else self.todo_list
        list_label = "Urgent" if is_urgent else "To Do"
        due_iso = self._compute_due(severity)

        fields: dict[str, Any] = {
            "idList": list_id,
            "name": title,
            "desc": desc,
            "pos": "top",
        }
        if due_iso:
            fields["due"] = due_iso

        async with httpx.AsyncClient(
            base_url=API, headers=self.headers, timeout=TIMEOUT_SECONDS, transport=self.transport
        ) as client:
            resp = await client.post("/cards", json=fields)
            resp.raise_for_status()
            data = resp.json()

        return TrelloCardResult(
            card_id=data["id"],
            card_url=data.get("shortUrl") or data.get("url") or f"https://trello.com/c/{data['id']}",
            list_name=list_label,
            due_iso=due_iso,
        )

    async def add_comment(self, card_id: str, text: str) -> None:
        """Appends a timestamped activity comment to an existing Trello work-order card."""
        if not card_id or not text.strip():
            return
        async with httpx.AsyncClient(
            base_url=API, headers=self.headers, timeout=TIMEOUT_SECONDS, transport=self.transport
        ) as client:
            resp = await client.post(f"/cards/{card_id}/actions/comments", params={"text": text.strip()})
            resp.raise_for_status()

    async def update_status(self, card_id: str, status: str, note: str | None = None) -> None:
        """Syncs a ticket status update (/progress or /resolve) to the Trello card."""
        if not card_id:
            return
        status_lower = (status or "").strip().lower()
        target_list = None
        if status_lower == "in progress" and self.in_progress_list:
            target_list = self.in_progress_list
        elif status_lower == "resolved" and self.done_list:
            target_list = self.done_list

        comment = f"Status updated to **{status}**" + (f" — Note: {note}" if note else "")
        async with httpx.AsyncClient(
            base_url=API, headers=self.headers, timeout=TIMEOUT_SECONDS, transport=self.transport
        ) as client:
            if target_list:
                r_move = await client.put(f"/cards/{card_id}", json={"idList": target_list})
                r_move.raise_for_status()
            r_cmt = await client.post(f"/cards/{card_id}/actions/comments", params={"text": comment})
            r_cmt.raise_for_status()

    def update_status_sync(self, card_id: str, status: str, note: str | None = None) -> bool:
        """Synchronous status sync helper for the Streamlit staff dashboard."""
        if not card_id:
            return False
        status_lower = (status or "").strip().lower()
        target_list = None
        if status_lower == "in progress" and self.in_progress_list:
            target_list = self.in_progress_list
        elif status_lower == "resolved" and self.done_list:
            target_list = self.done_list

        comment = f"Status updated via Staff Dashboard to **{status}**" + (f" — Note: {note}" if note else "")
        try:
            with httpx.Client(
                base_url=API, headers=self.headers, timeout=TIMEOUT_SECONDS, transport=self.sync_transport
            ) as client:
                if target_list:
                    r_move = client.put(f"/cards/{card_id}", json={"idList": target_list})
                    r_move.raise_for_status()
                r_cmt = client.post(f"/cards/{card_id}/actions/comments", params={"text": comment})
                r_cmt.raise_for_status()
            return True
        except Exception as exc:
            logger.warning("Trello status sync failed for card %s: %s", card_id, exc)
            return False


_SINGLETON: TrelloWorkOrders | None = None
_INIT_ATTEMPTED = False


def get_trello_client() -> TrelloWorkOrders | None:
    """Lazy singleton accessor. Never crashes the bot if Trello isn't configured or unreachable."""
    global _SINGLETON, _INIT_ATTEMPTED
    if _INIT_ATTEMPTED:
        return _SINGLETON
    _INIT_ATTEMPTED = True
    try:
        _SINGLETON = TrelloWorkOrders.from_env()
        if _SINGLETON:
            logger.info("Trello work-order dispatch enabled.")
    except Exception as exc:
        logger.warning("Trello integration disabled (%s).", exc)
        _SINGLETON = None
    return _SINGLETON


def reset_trello_client() -> None:
    global _SINGLETON, _INIT_ATTEMPTED
    _SINGLETON = None
    _INIT_ATTEMPTED = False
