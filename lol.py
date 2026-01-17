#!/usr/bin/env python3
"""
Annoyotron v3.1 — "Wake-only" mode + persistent "Active Assistant" window for meetup management.

✅ What you asked for (Bot behavior "3"):
- The bot **only talks when addressed** (mention / reply-to-bot / wake phrase).
- Once addressed, it enters **ACTIVE mode** for that chat:
    - While ACTIVE, it will keep monitoring the conversation and (when relevant) use the LLM to:
      - start a meetup poll
      - add/suggest new date/time options
      - reschedule/change details
      - confirm/cancel
    - ACTIVE mode persists until:
      - /dismiss (or /sleep), OR
      - active window expires (default 30 min since last wake)
- Uses Telegram **inline keyboards** for the availability poll + location picker (“widgets”).
- Persistent SQLite storage: chats, members, memory messages, meetup sessions/options/votes.
- Admin-only switch, mute timers, quiet hours, autopoke (optional), robust callback payloads.
- Debug logs to terminal that show exactly why the bot did/didn’t act.

Requirements:
  pip install -U python-telegram-bot python-dotenv openai

.env:
  TELEGRAM_BOT_TOKEN=...
  OPENAI_API_KEY=...
  OPENAI_MODEL=gpt-4.1-mini
  BOT_DB_PATH=bot_state.sqlite3
  BOT_TZ=Asia/Singapore

Optional knobs:
  DEBUG_LOG=1
  ACTIVE_WINDOW_MIN=30
  QUIET_HOUR_START=1
  QUIET_HOUR_END=9
  CHAT_REPLY_COOLDOWN_SEC=20
  CHAT_MAX_REPLIES_PER_10MIN=12
  USER_REPLY_COOLDOWN_SEC=60
  MAX_MEMORY_MESSAGES=20
  PERSIST_MEMORY_MESSAGES=80
  DEFAULT_MUTE_MIN=30
  AUTOPOKE_DEFAULT_INTERVAL_MIN=180
  AUTOPOKE_MIN_SILENCE_MIN=30
  MEETUP_DEFAULT_NAG_INTERVAL_MIN=5
  MEETUP_MAX_MENTIONS=5
  MEETUP_VOTE_THRESHOLD=3
  MEETUP_REMINDER_MINUTES=30

BotFather:
  - Disable privacy if you want it to receive all group messages:
    BotFather -> /setprivacy -> Disable
"""

import os
import re
import time
import json
import asyncio
import sqlite3
import logging
import secrets
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Optional, Dict, List, Tuple

from dotenv import load_dotenv

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ChatType
from telegram.error import TimedOut, NetworkError, BadRequest
from telegram.request import HTTPXRequest
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)

# ============================================================
# Load env
# ============================================================
load_dotenv()

# ============================================================
# Logging
# ============================================================
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("annoyotron")

DEBUG_LOG = (os.getenv("DEBUG_LOG") or "1").strip() == "1"

def dlog(msg: str, **kv):
    if not DEBUG_LOG:
        return
    payload = " ".join([f"{k}={v}" for k, v in kv.items()])
    logger.info("[DBG] %s%s", msg, (" | " + payload) if payload else "")

# ============================================================
# Env vars
# ============================================================
TELEGRAM_BOT_TOKEN = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("Missing TELEGRAM_BOT_TOKEN. Put it in .env")

OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()
USE_OPENAI = bool(OPENAI_API_KEY)
OPENAI_MODEL = os.getenv("OPENAI_MODEL") or "gpt-4.1-mini"

DB_PATH = (os.getenv("BOT_DB_PATH") or "bot_state.sqlite3").strip()
BOT_TZ = os.getenv("BOT_TZ") or "Asia/Singapore"

ACTIVE_WINDOW_MIN = int(os.getenv("ACTIVE_WINDOW_MIN") or 30)

# ============================================================
# Optional OpenAI (Responses API)
# ============================================================
oai_client = None
if USE_OPENAI:
    try:
        from openai import OpenAI
        oai_client = OpenAI(api_key=OPENAI_API_KEY)
        logger.info("OpenAI enabled (Responses API), model=%s", OPENAI_MODEL)
    except Exception as e:
        logger.warning("OpenAI disabled: %s", e)
        USE_OPENAI = False
        oai_client = None

# ============================================================
# Behavior knobs (env-configurable)
# ============================================================
CHAT_REPLY_COOLDOWN_SEC = int(os.getenv("CHAT_REPLY_COOLDOWN_SEC") or 20)
CHAT_MAX_REPLIES_PER_10MIN = int(os.getenv("CHAT_MAX_REPLIES_PER_10MIN") or 12)
USER_REPLY_COOLDOWN_SEC = int(os.getenv("USER_REPLY_COOLDOWN_SEC") or 60)

QUIET_HOUR_START = int(os.getenv("QUIET_HOUR_START") or 1)
QUIET_HOUR_END = int(os.getenv("QUIET_HOUR_END") or 9)

MAX_MEMORY_MESSAGES = int(os.getenv("MAX_MEMORY_MESSAGES") or 20)
PERSIST_MEMORY_MESSAGES = int(os.getenv("PERSIST_MEMORY_MESSAGES") or 80)

DEFAULT_MUTE_MIN = int(os.getenv("DEFAULT_MUTE_MIN") or 30)

AUTOPOKE_DEFAULT_INTERVAL_MIN = int(os.getenv("AUTOPOKE_DEFAULT_INTERVAL_MIN") or 180)
AUTOPOKE_MIN_SILENCE_MIN = int(os.getenv("AUTOPOKE_MIN_SILENCE_MIN") or 30)

MEETUP_DEFAULT_NAG_INTERVAL_MIN = int(os.getenv("MEETUP_DEFAULT_NAG_INTERVAL_MIN") or 5)
MEETUP_MAX_MENTIONS = int(os.getenv("MEETUP_MAX_MENTIONS") or 5)
MEETUP_VOTE_THRESHOLD = int(os.getenv("MEETUP_VOTE_THRESHOLD") or 3)
MEETUP_REMINDER_MINUTES = int(os.getenv("MEETUP_REMINDER_MINUTES") or 30)

MUTE_KEYWORDS = ("stop", "stfu", "shut up", "mute bot", "no bot", "quiet bot", "chill")
DISMISS_KEYWORDS = ("dismiss", "sleep", "go away", "stop listening", "enough bot", "ok bot stop")

# “wake phrases”: addressing the bot in natural text; we ONLY wake on these / mentions / reply-to-bot
WAKE_WORDS = ("bot", "annoyotron", "oi bot")

# Scheduling triggers (used for deciding when to call meetup intent while ACTIVE or if an active meetup exists)
TRIGGER_WORDS = (
    "meet", "meetup", "lunch", "dinner", "supper", "where eat", "hungry",
    "what time", "what day", "tmr", "tomorrow", "today", "tonight",
    "change time", "change timing", "reschedule", "postpone", "bring forward",
    "next week", "this week", "fri", "sat", "sun", "mon", "tue", "wed", "thu",
    "availability", "free", "busy", "can", "cannot", "cant"
)

CANTEENS = {
    "UTown": ["Fine Food", "Flavours", "Foodclique"],
    "Central": ["The Deck", "Frontier"],
    "SoC": ["The Terrace"],
    "Biz": ["The Pulse"],
    "Other": ["Techno Edge", "PGPR"],
}

# ============================================================
# DB helpers
# ============================================================
_db_lock = asyncio.Lock()

def _db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def _add_column_if_missing(conn: sqlite3.Connection, table: str, col_def: str):
    col_name = col_def.split()[0].strip()
    cols = [r["name"] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
    if col_name not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col_def}")

def init_db():
    conn = _db()
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS chats (
        chat_id INTEGER PRIMARY KEY,
        title TEXT,
        chaos_enabled INTEGER DEFAULT 0,
        roast_level INTEGER DEFAULT 2,
        autopoke_enabled INTEGER DEFAULT 0,
        autopoke_interval_min INTEGER DEFAULT 180,
        last_activity_ts INTEGER DEFAULT 0,
        muted_until_ts INTEGER DEFAULT 0,
        admin_only INTEGER DEFAULT 0,
        assistant_active_until_ts INTEGER DEFAULT 0
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS members (
        chat_id INTEGER,
        user_id INTEGER,
        username TEXT,
        first_name TEXT,
        roast_opt_in INTEGER DEFAULT 0,
        last_message_ts INTEGER DEFAULT 0,
        last_snippet TEXT DEFAULT '',
        PRIMARY KEY(chat_id, user_id)
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_id INTEGER,
        ts INTEGER,
        role TEXT,
        content TEXT
    )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_messages_chat_ts ON messages(chat_id, ts)")

    c.execute("""
    CREATE TABLE IF NOT EXISTS meetup_sessions (
        session_id TEXT PRIMARY KEY,
        chat_id INTEGER,
        status TEXT, -- 'collecting', 'confirmed', 'cancelled'
        title TEXT,
        location_area TEXT,
        location_canteen TEXT,
        chosen_option_id TEXT,
        created_by INTEGER,
        created_ts INTEGER,
        updated_ts INTEGER,
        nag_interval_min INTEGER DEFAULT 5,
        reminder_sent INTEGER DEFAULT 0
    )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_meetup_chat_status ON meetup_sessions(chat_id, status)")

    c.execute("""
    CREATE TABLE IF NOT EXISTS meetup_options (
        option_id TEXT PRIMARY KEY,
        session_id TEXT,
        start_ts INTEGER,
        label TEXT,
        created_ts INTEGER
    )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_options_session ON meetup_options(session_id, start_ts)")

    c.execute("""
    CREATE TABLE IF NOT EXISTS meetup_votes (
        session_id TEXT,
        option_id TEXT,
        user_id INTEGER,
        availability INTEGER, -- 1 yes, 0 no
        updated_ts INTEGER,
        PRIMARY KEY(session_id, option_id, user_id)
    )
    """)

    # Migrations (idempotent) for older DBs
    _add_column_if_missing(conn, "chats", "muted_until_ts INTEGER DEFAULT 0")
    _add_column_if_missing(conn, "chats", "admin_only INTEGER DEFAULT 0")
    _add_column_if_missing(conn, "chats", "assistant_active_until_ts INTEGER DEFAULT 0")
    _add_column_if_missing(conn, "meetup_sessions", "title TEXT")
    _add_column_if_missing(conn, "meetup_sessions", "reminder_sent INTEGER DEFAULT 0")

    conn.commit()
    conn.close()

async def db_exec(sql: str, params: Tuple = ()):
    async with _db_lock:
        def run():
            conn = _db()
            conn.execute(sql, params)
            conn.commit()
            conn.close()
        await asyncio.to_thread(run)

async def db_fetchone(sql: str, params: Tuple = ()):
    async with _db_lock:
        def run():
            conn = _db()
            cur = conn.execute(sql, params)
            row = cur.fetchone()
            conn.close()
            return row
        return await asyncio.to_thread(run)

async def db_fetchall(sql: str, params: Tuple = ()):
    async with _db_lock:
        def run():
            conn = _db()
            cur = conn.execute(sql, params)
            rows = cur.fetchall()
            conn.close()
            return rows
        return await asyncio.to_thread(run)

# ============================================================
# Utilities
# ============================================================
def now_ts() -> int:
    return int(time.time())

def tz() -> ZoneInfo:
    return ZoneInfo(BOT_TZ)

def local_now() -> datetime:
    return datetime.now(tz())

def clip(text: str, n: int = 180) -> str:
    t = (text or "").replace("\n", " ").strip()
    return t[:n] + ("…" if len(t) > n else "")

def in_quiet_hours() -> bool:
    h = local_now().hour
    if QUIET_HOUR_START < QUIET_HOUR_END:
        return QUIET_HOUR_START <= h < QUIET_HOUR_END
    return h >= QUIET_HOUR_START or h < QUIET_HOUR_END

def fmt_dt(ts_int: int) -> str:
    dt = datetime.fromtimestamp(int(ts_int), tz())
    return dt.strftime("%a %d %b, %H:%M")

def has_any_trigger(text: str) -> bool:
    tl = (text or "").lower()
    return any(t in tl for t in TRIGGER_WORDS)

# ============================================================
# Rate limiting (chat + per-user)
# ============================================================
chat_last_reply_ts: Dict[int, int] = {}
chat_reply_times: Dict[int, List[int]] = {}
user_last_reply_ts: Dict[Tuple[int, int], int] = {}

def should_rate_limit(chat_id: int, user_id: int) -> bool:
    now = now_ts()
    last_chat = chat_last_reply_ts.get(chat_id, 0)
    if now - last_chat < CHAT_REPLY_COOLDOWN_SEC:
        return True

    key = (chat_id, user_id)
    last_user = user_last_reply_ts.get(key, 0)
    if now - last_user < USER_REPLY_COOLDOWN_SEC:
        return True

    times = chat_reply_times.setdefault(chat_id, [])
    cutoff = now - 600
    while times and times[0] < cutoff:
        times.pop(0)
    if len(times) >= CHAT_MAX_REPLIES_PER_10MIN:
        return True
    return False

def mark_replied(chat_id: int, user_id: int):
    now = now_ts()
    chat_last_reply_ts[chat_id] = now
    user_last_reply_ts[(chat_id, user_id)] = now
    chat_reply_times.setdefault(chat_id, []).append(now)

# ============================================================
# Persistent memory for LLM context
# ============================================================
async def remember(chat_id: int, role: str, content: str):
    content = clip(content, 500)
    await db_exec(
        "INSERT INTO messages(chat_id, ts, role, content) VALUES (?, ?, ?, ?)",
        (chat_id, now_ts(), role, content),
    )
    await db_exec(
        """
        DELETE FROM messages
        WHERE id IN (
            SELECT id FROM messages
            WHERE chat_id = ?
            ORDER BY ts DESC
            LIMIT -1 OFFSET ?
        )
        """,
        (chat_id, PERSIST_MEMORY_MESSAGES),
    )

async def get_memory(chat_id: int) -> List[Dict[str, str]]:
    rows = await db_fetchall(
        "SELECT role, content FROM messages WHERE chat_id=? ORDER BY ts ASC",
        (chat_id,),
    )
    trimmed = rows[-MAX_MEMORY_MESSAGES:]
    return [{"role": r["role"], "content": r["content"]} for r in trimmed]

# ============================================================
# Telegram send safety
# ============================================================
async def safe_reply(msg, text: str, **kwargs):
    try:
        return await msg.reply_text(text, **kwargs)
    except (TimedOut, NetworkError) as e:
        logger.warning("Telegram reply failed (network): %s", e)
        return None
    except Exception as e:
        logger.warning("Telegram reply failed: %s", e)
        return None

async def safe_send(bot, chat_id: int, text: str, **kwargs):
    try:
        return await bot.send_message(chat_id=chat_id, text=text, **kwargs)
    except (TimedOut, NetworkError) as e:
        logger.warning("Telegram send failed (network): %s", e)
        return None
    except Exception as e:
        logger.warning("Telegram send failed: %s", e)
        return None

async def safe_edit_query_message(query, text: str, **kwargs):
    try:
        return await query.edit_message_text(text, **kwargs)
    except BadRequest as e:
        # Telegram throws if content+markup are identical
        if "Message is not modified" in str(e):
            dlog("EDIT_SKIPPED_NOT_MODIFIED")
            return None
        raise

# ============================================================
# Chat/member helpers + admin/mute gate
# ============================================================
async def ensure_chat_row(chat_id: int, title: str):
    await db_exec("""
        INSERT INTO chats(chat_id, title, last_activity_ts)
        VALUES(?, ?, ?)
        ON CONFLICT(chat_id) DO UPDATE SET title=excluded.title
    """, (chat_id, title or "", now_ts()))

async def touch_chat_activity(chat_id: int):
    await db_exec("UPDATE chats SET last_activity_ts=? WHERE chat_id=?", (now_ts(), chat_id))

async def ensure_member_row(chat_id: int, user):
    await db_exec("""
        INSERT INTO members(chat_id, user_id, username, first_name, last_message_ts, last_snippet)
        VALUES(?, ?, ?, ?, ?, ?)
        ON CONFLICT(chat_id, user_id) DO UPDATE SET
            username=excluded.username,
            first_name=excluded.first_name,
            last_message_ts=excluded.last_message_ts,
            last_snippet=excluded.last_snippet
    """, (chat_id, user.id, user.username, user.first_name or "", now_ts(), ""))

async def update_member_snippet(chat_id: int, user_id: int, snippet: str):
    await db_exec(
        "UPDATE members SET last_message_ts=?, last_snippet=? WHERE chat_id=? AND user_id=?",
        (now_ts(), clip(snippet, 180), chat_id, user_id),
    )

async def is_admin(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    try:
        chat = update.effective_chat
        user = update.effective_user
        if not chat or not user:
            return False
        member = await context.bot.get_chat_member(chat.id, user.id)
        return member.status in ("administrator", "creator")
    except Exception:
        return False

async def admin_gate(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    chat = update.effective_chat
    if not chat:
        return False
    row = await db_fetchone("SELECT admin_only FROM chats WHERE chat_id=?", (chat.id,))
    admin_only = int(row["admin_only"] or 0) if row else 0
    if not admin_only:
        return True
    if await is_admin(update, context):
        return True
    await safe_reply(update.effective_message, "⛔ Admin-only mode is ON. Ask an admin to do that.")
    return False

async def is_muted(chat_id: int) -> bool:
    row = await db_fetchone("SELECT muted_until_ts FROM chats WHERE chat_id=?", (chat_id,))
    if not row:
        return False
    return now_ts() < int(row["muted_until_ts"] or 0)

async def mute_chat(chat_id: int, minutes: int):
    until = now_ts() + max(1, minutes) * 60
    await db_exec("UPDATE chats SET muted_until_ts=? WHERE chat_id=?", (until, chat_id))

async def unmute_chat(chat_id: int):
    await db_exec("UPDATE chats SET muted_until_ts=0 WHERE chat_id=?", (chat_id,))

# ============================================================
# "Active Assistant" wake window (core request)
# ============================================================
async def set_active(chat_id: int, minutes: int = ACTIVE_WINDOW_MIN):
    until = now_ts() + max(1, minutes) * 60
    await db_exec("UPDATE chats SET assistant_active_until_ts=? WHERE chat_id=?", (until, chat_id))
    dlog("ACTIVE_SET", chat_id=chat_id, until=until)

async def clear_active(chat_id: int):
    await db_exec("UPDATE chats SET assistant_active_until_ts=0 WHERE chat_id=?", (chat_id,))
    dlog("ACTIVE_CLEARED", chat_id=chat_id)

async def is_active(chat_id: int) -> bool:
    row = await db_fetchone("SELECT assistant_active_until_ts FROM chats WHERE chat_id=?", (chat_id,))
    if not row:
        return False
    return now_ts() < int(row["assistant_active_until_ts"] or 0)

# ============================================================
# Mention detection via entities
# ============================================================
def is_bot_mentioned(msg, bot_username: str) -> bool:
    if not msg or not bot_username:
        return False
    uname = bot_username.lower()
    text = (msg.text or "") + (msg.caption or "")
    entities = list(msg.entities or []) + list(msg.caption_entities or [])
    for e in entities:
        if e.type == "mention":
            seg = text[e.offset:e.offset + e.length].lower()
            if seg == f"@{uname}":
                return True
    return f"@{uname}" in text.lower()

def wake_phrase_present(text: str) -> bool:
    """Wake phrase anywhere, but as a word boundary (avoid 'robot')."""
    tl = (text or "").lower()
    for w in WAKE_WORDS:
        # e.g. "bot" as separate word, or "oi bot"
        if re.search(rf"\b{re.escape(w)}\b", tl):
            return True
    return False

# ============================================================
# OpenAI — persona + reply + meetup intent extraction
# ============================================================
def build_persona(roast_level: int) -> str:
    return f"""
You are ANNOYOTRON, a chaotic roasty group-chat bro with Singapore/NUS vibes ("bro", "lah", "eh").
Roast level: {roast_level}/5.

Style:
- Short, punchy, witty. 1–3 sentences.
- Light teasing, mock disbelief, dramatic reactions, "skill issue" energy.
- Use Singlish particles sometimes (lah, sia, leh), but not every line.

Rules (must follow):
- No slurs, no identity-based insults, no threats, no sexual content.
- No humiliation or sustained targeting of one person.
- If someone says stop/mute/chill, back off immediately.
- If asked for help, you can still be funny but give a useful hint.
"""

async def llm_chat_reply(chat_id: int, roast_level: int, user_name: str, text: str) -> Optional[str]:
    if not USE_OPENAI or not oai_client:
        return None

    system = build_persona(roast_level)
    memory = await get_memory(chat_id)
    items = [{"role": "system", "content": system}]
    items.extend(memory)
    items.append({"role": "user", "content": f"{user_name}: {text}"})

    def _call():
        resp = oai_client.responses.create(
            model=OPENAI_MODEL,
            input=items,
            max_output_tokens=180,
        )
        out = []
        for item in resp.output:
            if item.type == "message":
                for c in item.content:
                    if c.type == "output_text":
                        out.append(c.text)
        return " ".join(out).strip() or None

    try:
        return await asyncio.to_thread(_call)
    except Exception as e:
        logger.warning("LLM chat call failed: %s", e)
        return None

MEETUP_INTENT_SYSTEM = f"""
You are a strict JSON planner for meetup scheduling in a Telegram group chat.
Timezone is {BOT_TZ}. Return ONLY valid JSON, no markdown.

Choose action:
- "NONE"
- "START_MEETUP"
- "ADD_OPTIONS"
- "SUGGEST_OPTIONS"
- "CHANGE_DETAILS"
- "CANCEL_MEETUP"
- "CONFIRM_MEETUP"
- "SET_LOCATION"
- "ASK_STATUS"

Optional fields:
- action
- title: short label like "lunch" or "dinner" or "meetup"
- area: one of ["UTown","Central","SoC","Biz","Other"] or null
- canteen: string or null
- datetime_suggestions: array of ISO strings local time in {BOT_TZ}, e.g. "2026-01-18T12:30"
- note: short reason

Rules:
- If the chat is clearly discussing meeting up or rescheduling, choose an action.
- If active meetup exists and user wants to change timing/date, output CHANGE_DETAILS with 2–6 datetime_suggestions if possible.
- If asked for status, ASK_STATUS.
- If asked to cancel, CANCEL_MEETUP.
- If asked to confirm/lock in, CONFIRM_MEETUP.
- Otherwise NONE.
"""

async def llm_meetup_intent(chat_id: int, user_text: str, active_session_summary: str) -> Dict:
    if not USE_OPENAI or not oai_client:
        return {"action": "NONE"}

    prompt = {
        "chat_context": active_session_summary,
        "user_message": user_text,
        "now_local": local_now().strftime("%Y-%m-%dT%H:%M"),
        "timezone": BOT_TZ,
    }

    def _call():
        resp = oai_client.responses.create(
            model=OPENAI_MODEL,
            input=[
                {"role": "system", "content": MEETUP_INTENT_SYSTEM},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
            max_output_tokens=220,
        )
        out = []
        for item in resp.output:
            if item.type == "message":
                for c in item.content:
                    if c.type == "output_text":
                        out.append(c.text)
        txt = " ".join(out).strip()
        try:
            return json.loads(txt)
        except Exception:
            logger.warning("Meetup intent not valid JSON: %s", txt[:300])
            return {"action": "NONE"}

    try:
        return await asyncio.to_thread(_call)
    except Exception as e:
        logger.warning("Meetup intent LLM error: %s", e)
        return {"action": "NONE"}

# ============================================================
# Meetup engine (availability poll + reminders + chat-driven changes)
# ============================================================
@dataclass
class MeetupSession:
    session_id: str
    chat_id: int
    status: str
    title: str
    area: Optional[str]
    canteen: Optional[str]
    chosen_option_id: Optional[str]
    nag_interval_min: int
    reminder_sent: int

async def get_active_meetup(chat_id: int) -> Optional[MeetupSession]:
    row = await db_fetchone(
        "SELECT * FROM meetup_sessions WHERE chat_id=? AND status NOT IN ('confirmed','cancelled') ORDER BY created_ts DESC LIMIT 1",
        (chat_id,),
    )
    if not row:
        return None
    return MeetupSession(
        session_id=row["session_id"],
        chat_id=int(row["chat_id"]),
        status=row["status"],
        title=row["title"] or "meetup",
        area=row["location_area"],
        canteen=row["location_canteen"],
        chosen_option_id=row["chosen_option_id"],
        nag_interval_min=int(row["nag_interval_min"] or MEETUP_DEFAULT_NAG_INTERVAL_MIN),
        reminder_sent=int(row["reminder_sent"] or 0),
    )

async def describe_meetup_for_llm(chat_id: int) -> str:
    sess = await get_active_meetup(chat_id)
    if not sess:
        return "No active meetup."
    options = await db_fetchall(
        "SELECT option_id, start_ts, label FROM meetup_options WHERE session_id=? ORDER BY start_ts ASC LIMIT 8",
        (sess.session_id,),
    )
    opt_lines = [f"- {o['option_id']}: {fmt_dt(o['start_ts'])}" for o in options]
    loc = f"{sess.area or 'TBD'} / {sess.canteen or 'TBD'}"
    return (
            f"Active meetup:\n"
            f"session_id={sess.session_id}\n"
            f"title={sess.title}\n"
            f"status={sess.status}\n"
            f"location={loc}\n"
            f"options:\n" + ("\n".join(opt_lines) if opt_lines else "- (none yet)")
    )

def _mk_cb(session_id: str, action: str, a: str = "", b: str = "") -> str:
    s = "|".join(["m", session_id, action, a, b])
    return s[:64]

async def create_meetup(chat_id: int, created_by: int, title: str, nag_interval: int):
    sid = secrets.token_hex(4)
    await db_exec(
        """
        INSERT INTO meetup_sessions(session_id, chat_id, status, title, created_by, created_ts, updated_ts, nag_interval_min)
        VALUES (?, ?, 'collecting', ?, ?, ?, ?, ?)
        """,
        (sid, chat_id, title or "meetup", created_by, now_ts(), now_ts(), nag_interval),
    )
    return sid

def _default_candidate_times() -> List[datetime]:
    base = local_now().replace(second=0, microsecond=0)
    days = [base.date(), (base + timedelta(days=1)).date(), (base + timedelta(days=2)).date()]
    slots = [(12, 0), (12, 30), (13, 0), (18, 0), (18, 30), (19, 0)]
    out = []
    for d in days:
        for hh, mm in slots:
            dt = datetime(d.year, d.month, d.day, hh, mm, tzinfo=tz())
            if dt > base + timedelta(minutes=10):
                out.append(dt)
    return out[:10]

def _parse_iso_local(s: str) -> Optional[int]:
    try:
        dt = datetime.strptime(s.strip(), "%Y-%m-%dT%H:%M").replace(tzinfo=tz())
        return int(dt.timestamp())
    except Exception:
        return None

async def upsert_options(session_id: str, iso_list: List[str], max_add: int = 10) -> int:
    added = 0
    for s in iso_list[:max_add]:
        ts_int = _parse_iso_local(s)
        if not ts_int:
            continue
        exists = await db_fetchone(
            "SELECT option_id FROM meetup_options WHERE session_id=? AND start_ts=?",
            (session_id, ts_int),
        )
        if exists:
            continue
        oid = secrets.token_hex(3)
        await db_exec(
            "INSERT INTO meetup_options(option_id, session_id, start_ts, label, created_ts) VALUES (?,?,?,?,?)",
            (oid, session_id, ts_int, "", now_ts()),
        )
        added += 1
    return added

async def compute_option_scores(session_id: str) -> List[Tuple[str, int, int]]:
    rows = await db_fetchall(
        """
        SELECT o.option_id,
               COALESCE(SUM(CASE WHEN v.availability=1 THEN 1 ELSE 0 END), 0) AS yes_cnt,
               COALESCE(COUNT(v.user_id), 0) AS total_cnt
        FROM meetup_options o
        LEFT JOIN meetup_votes v
          ON v.session_id=o.session_id AND v.option_id=o.option_id
        WHERE o.session_id=?
        GROUP BY o.option_id
        ORDER BY yes_cnt DESC, total_cnt DESC
        """,
        (session_id,),
    )
    return [(r["option_id"], int(r["yes_cnt"]), int(r["total_cnt"])) for r in rows]

async def pick_best_option(session_id: str) -> Optional[str]:
    scores = await compute_option_scores(session_id)
    if not scores:
        return None
    for oid, yes_cnt, _ in scores:
        if yes_cnt >= MEETUP_VOTE_THRESHOLD:
            return oid
    return scores[0][0]

async def render_meetup_poll_message(session_id: str) -> Tuple[str, InlineKeyboardMarkup]:
    sess_row = await db_fetchone("SELECT * FROM meetup_sessions WHERE session_id=?", (session_id,))
    if not sess_row:
        return ("Meetup not found.", InlineKeyboardMarkup([]))

    title = sess_row["title"] or "meetup"
    area = sess_row["location_area"] or "TBD"
    canteen = sess_row["location_canteen"] or "TBD"

    options = await db_fetchall(
        "SELECT option_id, start_ts FROM meetup_options WHERE session_id=? ORDER BY start_ts ASC LIMIT 12",
        (session_id,),
    )
    scores = {oid: (yes, tot) for oid, yes, tot in await compute_option_scores(session_id)}

    lines = [
        f"📊 **{title.upper()} POLL**",
        f"📍 Location: **{area} / {canteen}**",
        f"🗳️ Mark times you **can make it** (✅).",
        f"Threshold: **{MEETUP_VOTE_THRESHOLD}** ✅ to suggest a best slot.",
        "",
        "**Options:**"
    ]

    kb: List[List[InlineKeyboardButton]] = []
    for o in options:
        oid = o["option_id"]
        yes, tot = scores.get(oid, (0, 0))
        lines.append(f"- {fmt_dt(o['start_ts'])}  ✅{yes}/{MEETUP_VOTE_THRESHOLD} (total:{tot})")
        kb.append([
            InlineKeyboardButton(f"✅ {fmt_dt(o['start_ts'])}", callback_data=_mk_cb(session_id, "yes", oid, "")),
            InlineKeyboardButton("❌", callback_data=_mk_cb(session_id, "no", oid, "")),
        ])

    kb.append([
        InlineKeyboardButton("📍 Set location", callback_data=_mk_cb(session_id, "loc", "", "")),
        InlineKeyboardButton("🔄 Suggest new times", callback_data=_mk_cb(session_id, "suggest", "", "")),
    ])
    kb.append([
        InlineKeyboardButton("✅ Finalize best slot", callback_data=_mk_cb(session_id, "finalize", "", "")),
        InlineKeyboardButton("🛑 Cancel", callback_data=_mk_cb(session_id, "cancel", "", "")),
    ])

    return "\n".join(lines), InlineKeyboardMarkup(kb)

async def finalize_meetup(application: Application, chat_id: int, session_id: str) -> Optional[str]:
    sess_row = await db_fetchone("SELECT * FROM meetup_sessions WHERE session_id=?", (session_id,))
    if not sess_row:
        return None
    if sess_row["status"] in ("confirmed", "cancelled"):
        return None

    best = await pick_best_option(session_id)
    if not best:
        return None

    await db_exec(
        "UPDATE meetup_sessions SET status='confirmed', chosen_option_id=?, updated_ts=? WHERE session_id=?",
        (best, now_ts(), session_id),
    )

    opt = await db_fetchone("SELECT start_ts FROM meetup_options WHERE option_id=? AND session_id=?", (best, session_id))
    if opt:
        await schedule_meetup_reminder(application, chat_id, session_id, int(opt["start_ts"]))
    return best

# ============================================================
# Jobs: meetup nag + meetup reminder (autopoke left out by default; add back if needed)
# ============================================================
_meetup_nag_jobs: Dict[str, object] = {}
_meetup_reminder_jobs: Dict[str, object] = {}

async def meetup_nag_job(context: ContextTypes.DEFAULT_TYPE):
    chat_id = context.job.chat_id
    session_id = (context.job.data or {}).get("session_id")
    if not session_id:
        return
    if in_quiet_hours():
        return

    sess = await db_fetchone("SELECT * FROM meetup_sessions WHERE session_id=?", (session_id,))
    if not sess or sess["status"] in ("confirmed", "cancelled"):
        cancel_meetup_nag(session_id)
        return

    all_members = await db_fetchall("SELECT user_id, username, first_name FROM members WHERE chat_id=?", (chat_id,))
    if not all_members:
        return

    voted = await db_fetchall("SELECT DISTINCT user_id FROM meetup_votes WHERE session_id=?", (session_id,))
    voted_ids = {r["user_id"] for r in voted}
    non_voters = [m for m in all_members if m["user_id"] not in voted_ids]

    if not non_voters:
        cancel_meetup_nag(session_id)
        return

    mentions = []
    for m in non_voters[:MEETUP_MAX_MENTIONS]:
        if m["username"]:
            mentions.append(f"@{m['username']}")
        else:
            mentions.append(m["first_name"] or "someone")

    text, _ = await render_meetup_poll_message(session_id)
    nag_text = f"Oi {', '.join(mentions)}! Vote your availability leh. 👇\n\n" + text.split("\n", 1)[0]
    await safe_send(context.bot, chat_id, nag_text)

def schedule_meetup_nag(application: Application, chat_id: int, session_id: str, nag_interval_min: int):
    old = _meetup_nag_jobs.get(session_id)
    if old is not None:
        try: old.schedule_removal()
        except Exception: pass
        _meetup_nag_jobs.pop(session_id, None)

    job = application.job_queue.run_repeating(
        meetup_nag_job,
        interval=max(1, nag_interval_min) * 60,
        first=max(60, nag_interval_min * 30),
        chat_id=chat_id,
        name=f"meetup_nag_{chat_id}_{session_id}",
        data={"session_id": session_id},
    )
    _meetup_nag_jobs[session_id] = job
    dlog("NAG_SCHEDULED", chat_id=chat_id, session_id=session_id, interval_min=nag_interval_min)

def cancel_meetup_nag(session_id: str):
    job = _meetup_nag_jobs.get(session_id)
    if job is not None:
        try: job.schedule_removal()
        except Exception: pass
        _meetup_nag_jobs.pop(session_id, None)
        dlog("NAG_CANCELLED", session_id=session_id)

async def meetup_reminder_job(context: ContextTypes.DEFAULT_TYPE):
    chat_id = context.job.chat_id
    session_id = (context.job.data or {}).get("session_id")
    if not session_id:
        return

    sess = await db_fetchone("SELECT * FROM meetup_sessions WHERE session_id=?", (session_id,))
    if not sess or sess["status"] != "confirmed":
        return
    if int(sess["reminder_sent"] or 0) == 1:
        return

    chosen = sess["chosen_option_id"]
    opt = await db_fetchone("SELECT start_ts FROM meetup_options WHERE option_id=? AND session_id=?", (chosen, session_id))
    if not opt:
        return

    area = sess["location_area"] or "TBD"
    canteen = sess["location_canteen"] or "TBD"
    when = fmt_dt(opt["start_ts"])

    text = f"⏰ **MEETUP REMINDER**\n📍 {area} / {canteen}\n🕒 {when}\n\nDon’t be late lah 🏃"
    await safe_send(context.bot, chat_id, text, parse_mode="Markdown")
    await db_exec("UPDATE meetup_sessions SET reminder_sent=1 WHERE session_id=?", (session_id,))
    dlog("REMINDER_SENT", chat_id=chat_id, session_id=session_id)

async def schedule_meetup_reminder(application: Application, chat_id: int, session_id: str, start_ts: int):
    old = _meetup_reminder_jobs.get(session_id)
    if old is not None:
        try: old.schedule_removal()
        except Exception: pass
        _meetup_reminder_jobs.pop(session_id, None)

    now = local_now()
    meetup_dt = datetime.fromtimestamp(int(start_ts), tz())
    reminder_dt = meetup_dt - timedelta(minutes=MEETUP_REMINDER_MINUTES)
    delay = (reminder_dt - now).total_seconds()
    if delay <= 0:
        dlog("REMINDER_SKIP_PAST", chat_id=chat_id, session_id=session_id, delay=delay)
        return

    job = application.job_queue.run_once(
        meetup_reminder_job,
        when=delay,
        chat_id=chat_id,
        name=f"meetup_reminder_{chat_id}_{session_id}",
        data={"session_id": session_id},
    )
    _meetup_reminder_jobs[session_id] = job
    dlog("REMINDER_SCHEDULED", chat_id=chat_id, session_id=session_id, delay_s=int(delay))

# ============================================================
# Commands
# ============================================================
HELP_TEXT = (
    "🤖 Annoyotron commands:\n\n"
    "Wake me by mentioning @me / replying to me / saying 'bot' in a message.\n"
    "After waking, I’ll stay ACTIVE and help scheduling until /dismiss.\n\n"
    "/meetup – start an availability poll\n"
    "/meetup_status – show current meetup\n"
    "/meetup_cancel – cancel current meetup\n\n"
    "/dismiss – stop ACTIVE mode (I go quiet)\n"
    "/status – show settings\n"
    "/openai_test – test OpenAI\n"
)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await safe_reply(
        update.effective_message,
        "✅ Annoyotron online.\n"
        "If you want me to *receive all group messages*: BotFather → /setprivacy → Disable.\n\n"
        + HELP_TEXT
    )

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await safe_reply(update.effective_message, HELP_TEXT)

async def ping(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await safe_reply(update.effective_message, "pong ✅")

async def dismiss(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat
    await ensure_chat_row(chat.id, chat.title or "")
    await clear_active(chat.id)
    await safe_reply(update.effective_message, "😴 Ok I’ll go quiet. Mention me if you need scheduling again.")

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat
    await ensure_chat_row(chat.id, chat.title or "")
    row = await db_fetchone("SELECT * FROM chats WHERE chat_id=?", (chat.id,))
    if not row:
        await safe_reply(update.effective_message, "No settings yet.")
        return

    muted_until = int(row["muted_until_ts"] or 0)
    muted_str = "no"
    if muted_until and muted_until > now_ts():
        muted_str = datetime.fromtimestamp(muted_until, tz()).strftime("%Y-%m-%d %H:%M")

    active_until = int(row["assistant_active_until_ts"] or 0)
    active_str = "no"
    if active_until and active_until > now_ts():
        active_str = datetime.fromtimestamp(active_until, tz()).strftime("%Y-%m-%d %H:%M")

    await safe_reply(
        update.effective_message,
        f"Settings:\n"
        f"- active_until: {active_str}\n"
        f"- openai_enabled: {USE_OPENAI}\n"
        f"- openai_model: {OPENAI_MODEL}\n"
        f"- quiet_hours: {QUIET_HOUR_START}:00–{QUIET_HOUR_END}:00 ({BOT_TZ})\n"
        f"- muted_until: {muted_str}\n"
        f"- cooldown: chat {CHAT_REPLY_COOLDOWN_SEC}s, user {USER_REPLY_COOLDOWN_SEC}s\n"
        f"- cap: {CHAT_MAX_REPLIES_PER_10MIN} replies/10min"
    )

async def openai_test(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not USE_OPENAI or not oai_client:
        await safe_reply(update.effective_message, "OpenAI not available. Check OPENAI_API_KEY and `pip install -U openai`.")
        return

    def _call():
        resp = oai_client.responses.create(
            model=OPENAI_MODEL,
            input="Reply with exactly: OK",
            max_output_tokens=16,
        )
        out = []
        for item in resp.output:
            if item.type == "message":
                for c in item.content:
                    if c.type == "output_text":
                        out.append(c.text)
        return " ".join(out).strip()

    try:
        txt = await asyncio.to_thread(_call)
        await safe_reply(update.effective_message, f"OpenAI says: {txt}")
    except Exception as e:
        await safe_reply(update.effective_message, f"OpenAI test failed: {e}")

# ============================================================
# Meetup commands (manual start/status/cancel)
# ============================================================
async def meetup_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat
    user = update.effective_user
    await ensure_chat_row(chat.id, chat.title or "")

    if chat.type not in (ChatType.GROUP, ChatType.SUPERGROUP):
        await safe_reply(update.effective_message, "Meetup works best in a group chat lah.")
        return

    existing = await get_active_meetup(chat.id)
    if existing:
        await safe_reply(update.effective_message, "🚨 A meetup is already running. Use /meetup_status.")
        return

    nag_interval = MEETUP_DEFAULT_NAG_INTERVAL_MIN
    sid = await create_meetup(chat.id, user.id, "meetup", nag_interval)

    defaults = _default_candidate_times()
    iso = [d.strftime("%Y-%m-%dT%H:%M") for d in defaults]
    await upsert_options(sid, iso, max_add=10)

    text, markup = await render_meetup_poll_message(sid)
    await safe_send(context.bot, chat.id, text, reply_markup=markup, parse_mode="Markdown")
    schedule_meetup_nag(context.application, chat.id, sid, nag_interval)

    # wake + keep active while meetup ongoing
    await set_active(chat.id, ACTIVE_WINDOW_MIN)

async def meetup_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat
    sess = await get_active_meetup(chat.id)
    if not sess:
        await safe_reply(update.effective_message, "No active meetup. Use /meetup to start one.")
        return
    text, markup = await render_meetup_poll_message(sess.session_id)
    await safe_send(context.bot, chat.id, text, reply_markup=markup, parse_mode="Markdown")

async def meetup_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat
    sess = await get_active_meetup(chat.id)
    if not sess:
        await safe_reply(update.effective_message, "No active meetup to cancel.")
        return
    await db_exec("UPDATE meetup_sessions SET status='cancelled', updated_ts=? WHERE session_id=?", (now_ts(), sess.session_id))
    cancel_meetup_nag(sess.session_id)
    await safe_reply(update.effective_message, "🛑 Meetup cancelled. Y’all too chaotic sia.")

# ============================================================
# Meetup callback handler (inline “widgets”)
# ============================================================
async def meetup_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data or ""
    parts = data.split("|")
    if len(parts) < 3 or parts[0] != "m":
        return

    session_id = parts[1]
    action = parts[2]
    a = parts[3] if len(parts) > 3 else ""

    user = query.from_user
    chat_id = query.message.chat_id if query.message else None
    if not chat_id:
        return

    sess_row = await db_fetchone("SELECT * FROM meetup_sessions WHERE session_id=?", (session_id,))
    if not sess_row or int(sess_row["chat_id"]) != int(chat_id):
        await safe_edit_query_message(query, "This meetup is no longer active.")
        return
    if sess_row["status"] in ("confirmed", "cancelled"):
        await safe_edit_query_message(query, "This meetup is already closed.")
        return

    # Keep assistant ACTIVE after any interaction
    await set_active(chat_id, ACTIVE_WINDOW_MIN)

    if action in ("yes", "no"):
        option_id = a
        avail = 1 if action == "yes" else 0

        # Check if already set to avoid unnecessary updates
        existing = await db_fetchone(
            "SELECT availability FROM meetup_votes WHERE session_id=? AND option_id=? AND user_id=?",
            (session_id, option_id, user.id),
        )
        if existing and int(existing["availability"]) == avail:
            await query.answer("Already set 👍", show_alert=False)
            return

        await db_exec(
            """
            INSERT INTO meetup_votes(session_id, option_id, user_id, availability, updated_ts)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(session_id, option_id, user_id) DO UPDATE SET availability=excluded.availability, updated_ts=excluded.updated_ts
            """,
            (session_id, option_id, user.id, avail, now_ts()),
        )
        await db_exec("UPDATE meetup_sessions SET updated_ts=? WHERE session_id=?", (now_ts(), session_id))

        text, markup = await render_meetup_poll_message(session_id)
        best = await pick_best_option(session_id)
        if best:
            opt = await db_fetchone("SELECT start_ts FROM meetup_options WHERE option_id=? AND session_id=?", (best, session_id))
            if opt:
                scores = await compute_option_scores(session_id)
                yes_cnt = next((y for oid, y, _ in scores if oid == best), 0)
                if yes_cnt >= MEETUP_VOTE_THRESHOLD:
                    text += f"\n\n💡 **Suggestion:** **{fmt_dt(opt['start_ts'])}** is winning (✅{yes_cnt}). Hit **Finalize** to lock."
        await safe_edit_query_message(query, text, reply_markup=markup, parse_mode="Markdown")
        return

    if action == "cancel":
        await db_exec("UPDATE meetup_sessions SET status='cancelled', updated_ts=? WHERE session_id=?", (now_ts(), session_id))
        cancel_meetup_nag(session_id)
        await safe_edit_query_message(query, "🛑 Meetup cancelled.")
        return

    if action == "finalize":
        best = await finalize_meetup(context.application, chat_id, session_id)
        if not best:
            await query.answer("No options/votes yet.", show_alert=False)
            return
        sess2 = await db_fetchone("SELECT * FROM meetup_sessions WHERE session_id=?", (session_id,))
        opt = await db_fetchone("SELECT start_ts FROM meetup_options WHERE option_id=? AND session_id=?", (best, session_id))
        when = fmt_dt(opt["start_ts"]) if opt else "TBD"
        area = sess2["location_area"] or "TBD"
        canteen = sess2["location_canteen"] or "TBD"
        cancel_meetup_nag(session_id)
        await safe_edit_query_message(
            query,
            f"✅ **MEETUP CONFIRMED**\n📍 {area} / {canteen}\n🕒 {when}\n\nReminder will fire {MEETUP_REMINDER_MINUTES} min before.",
            parse_mode="Markdown",
        )
        return

    if action == "loc":
        kb = []
        row = []
        for area in CANTEENS.keys():
            row.append(InlineKeyboardButton(area, callback_data=_mk_cb(session_id, "set_area", area, "")))
            if len(row) == 2:
                kb.append(row); row = []
        if row:
            kb.append(row)
        kb.append([InlineKeyboardButton("Back", callback_data=_mk_cb(session_id, "back", "", ""))])
        await safe_edit_query_message(query, "📍 Pick an area:", reply_markup=InlineKeyboardMarkup(kb))
        return

    if action == "set_area":
        area = a
        if area not in CANTEENS:
            await query.answer("Invalid area", show_alert=False)
            return
        await db_exec("UPDATE meetup_sessions SET location_area=?, updated_ts=? WHERE session_id=?", (area, now_ts(), session_id))
        kb = []
        row = []
        for c in CANTEENS[area]:
            row.append(InlineKeyboardButton(c, callback_data=_mk_cb(session_id, "set_canteen", c, "")))
            if len(row) == 2:
                kb.append(row); row = []
        if row:
            kb.append(row)
        kb.append([InlineKeyboardButton("Back", callback_data=_mk_cb(session_id, "back", "", ""))])
        await safe_edit_query_message(query, f"📍 Area: **{area}**\nNow pick canteen:", reply_markup=InlineKeyboardMarkup(kb), parse_mode="Markdown")
        return

    if action == "set_canteen":
        canteen = a
        await db_exec("UPDATE meetup_sessions SET location_canteen=?, updated_ts=? WHERE session_id=?", (canteen, now_ts(), session_id))
        text, markup = await render_meetup_poll_message(session_id)
        await safe_edit_query_message(query, text, reply_markup=markup, parse_mode="Markdown")
        return

    if action == "suggest":
        summary = await describe_meetup_for_llm(chat_id)
        intent = await llm_meetup_intent(chat_id, "Suggest new times/options.", summary)
        iso_list = intent.get("datetime_suggestions") or []
        if not iso_list:
            iso_list = [d.strftime("%Y-%m-%dT%H:%M") for d in _default_candidate_times()[:6]]
        await upsert_options(session_id, iso_list, max_add=8)
        text, markup = await render_meetup_poll_message(session_id)
        await safe_edit_query_message(query, text, reply_markup=markup, parse_mode="Markdown")
        return

    if action == "back":
        text, markup = await render_meetup_poll_message(session_id)
        await safe_edit_query_message(query, text, reply_markup=markup, parse_mode="Markdown")
        return

# ============================================================
# Wake-only behavior + ACTIVE listening for meetup changes
# ============================================================
async def maybe_handle_meetup_from_chat(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str) -> bool:
    """
    While ACTIVE (or if a meetup exists), listen to conversation to adjust scheduling.
    Returns True if bot performed an explicit meetup action (and likely spoke).
    """
    chat = update.effective_chat
    msg = update.effective_message

    if chat.type not in (ChatType.GROUP, ChatType.SUPERGROUP):
        return False

    sess = await get_active_meetup(chat.id)
    active = await is_active(chat.id)

    # Only call LLM if:
    # - there is an active meetup, OR
    # - ACTIVE mode and message contains scheduling-ish triggers
    if not sess and not (active and has_any_trigger(text)):
        dlog("MEETUP_SKIP", chat_id=chat.id, active=active, has_meetup=bool(sess), trigger=has_any_trigger(text))
        return False

    # Quiet hours: if not explicitly woken, avoid actions (but if ACTIVE, you asked it to keep listening)
    # We'll allow meetup actions during ACTIVE even in quiet hours, but avoid spamming: require trigger.
    if in_quiet_hours() and not has_any_trigger(text):
        dlog("MEETUP_QUIET_SKIP_NO_TRIGGER", chat_id=chat.id)
        return False

    summary = await describe_meetup_for_llm(chat.id)
    intent = await llm_meetup_intent(chat.id, text, summary)
    action = (intent.get("action") or "NONE").upper()

    dlog(
        "MEETUP_INTENT",
        chat_id=chat.id,
        active=active,
        has_meetup=bool(sess),
        action=action,
        title=intent.get("title"),
        area=intent.get("area"),
        canteen=intent.get("canteen"),
        n_suggestions=len(intent.get("datetime_suggestions") or []),
    )

    if action == "NONE":
        return False

    # For meetup-affecting actions, require bot is ACTIVE OR message was a direct wake
    # (prevents the bot from changing plans silently when not in active assistant mode)
    if not active:
        dlog("MEETUP_BLOCK_NOT_ACTIVE", chat_id=chat.id, action=action)
        return False

    # Rate-limit speaking
    user_id = update.effective_user.id if update.effective_user else 0
    if should_rate_limit(chat.id, user_id):
        dlog("MEETUP_RETURN_RATE_LIMIT", chat_id=chat.id, user_id=user_id)
        return True  # we “handled” by deciding not to spam

    # Execute actions
    if action == "START_MEETUP":
        existing = await get_active_meetup(chat.id)
        if existing:
            await safe_reply(msg, "Meetup already running. Vote on the poll 👇")
            text2, markup = await render_meetup_poll_message(existing.session_id)
            await safe_send(context.bot, chat.id, text2, reply_markup=markup, parse_mode="Markdown")
            mark_replied(chat.id, user_id)
            return True

        sid = await create_meetup(chat.id, user_id, intent.get("title") or "meetup", MEETUP_DEFAULT_NAG_INTERVAL_MIN)

        iso_list = intent.get("datetime_suggestions") or []
        if not iso_list:
            iso_list = [d.strftime("%Y-%m-%dT%H:%M") for d in _default_candidate_times()[:8]]
        await upsert_options(sid, iso_list, max_add=10)

        area = intent.get("area")
        canteen = intent.get("canteen")
        if area and area in CANTEENS:
            await db_exec("UPDATE meetup_sessions SET location_area=?, updated_ts=? WHERE session_id=?", (area, now_ts(), sid))
        if canteen:
            await db_exec("UPDATE meetup_sessions SET location_canteen=?, updated_ts=? WHERE session_id=?", (canteen, now_ts(), sid))

        text2, markup = await render_meetup_poll_message(sid)
        await safe_reply(msg, "Ok I started a meetup poll 👇")
        await safe_send(context.bot, chat.id, text2, reply_markup=markup, parse_mode="Markdown")
        schedule_meetup_nag(context.application, chat.id, sid, MEETUP_DEFAULT_NAG_INTERVAL_MIN)
        mark_replied(chat.id, user_id)
        return True

    # Need existing meetup for most actions
    if not sess:
        await safe_reply(msg, "No active meetup. Say “bot start meetup” or use /meetup.")
        mark_replied(chat.id, user_id)
        return True

    if action == "ASK_STATUS":
        text2, markup = await render_meetup_poll_message(sess.session_id)
        await safe_reply(msg, "Here’s the current poll 👇")
        await safe_send(context.bot, chat.id, text2, reply_markup=markup, parse_mode="Markdown")
        mark_replied(chat.id, user_id)
        return True

    if action == "CANCEL_MEETUP":
        await db_exec("UPDATE meetup_sessions SET status='cancelled', updated_ts=? WHERE session_id=?", (now_ts(), sess.session_id))
        cancel_meetup_nag(sess.session_id)
        await safe_reply(msg, "Cancelled. Next time decide faster lah 😭")
        mark_replied(chat.id, user_id)
        return True

    if action in ("SET_LOCATION", "CHANGE_DETAILS"):
        did = False
        area = intent.get("area")
        canteen = intent.get("canteen")
        if area and area in CANTEENS:
            await db_exec("UPDATE meetup_sessions SET location_area=?, updated_ts=? WHERE session_id=?", (area, now_ts(), sess.session_id))
            did = True
        if canteen:
            await db_exec("UPDATE meetup_sessions SET location_canteen=?, updated_ts=? WHERE session_id=?", (canteen, now_ts(), sess.session_id))
            did = True

        iso_list = intent.get("datetime_suggestions") or []
        if iso_list:
            await upsert_options(sess.session_id, iso_list, max_add=10)
            did = True

        if did:
            text2, markup = await render_meetup_poll_message(sess.session_id)
            await safe_reply(msg, "Updated the poll based on chat. Vote again 👇")
            await safe_send(context.bot, chat.id, text2, reply_markup=markup, parse_mode="Markdown")
            mark_replied(chat.id, user_id)
            return True

        return False

    if action in ("ADD_OPTIONS", "SUGGEST_OPTIONS"):
        iso_list = intent.get("datetime_suggestions") or []
        if not iso_list:
            iso_list = [d.strftime("%Y-%m-%dT%H:%M") for d in _default_candidate_times()[:6]]
        added = await upsert_options(sess.session_id, iso_list, max_add=10)
        text2, markup = await render_meetup_poll_message(sess.session_id)
        await safe_reply(msg, f"Added {added} option(s). Vote here 👇")
        await safe_send(context.bot, chat.id, text2, reply_markup=markup, parse_mode="Markdown")
        mark_replied(chat.id, user_id)
        return True

    if action == "CONFIRM_MEETUP":
        best = await finalize_meetup(context.application, chat.id, sess.session_id)
        if not best:
            await safe_reply(msg, "Cannot confirm yet — no options/votes. Add times or vote first.")
            mark_replied(chat.id, user_id)
            return True
        sess2 = await db_fetchone("SELECT * FROM meetup_sessions WHERE session_id=?", (sess.session_id,))
        opt = await db_fetchone("SELECT start_ts FROM meetup_options WHERE option_id=? AND session_id=?", (best, sess.session_id))
        when = fmt_dt(opt["start_ts"]) if opt else "TBD"
        area = sess2["location_area"] or "TBD"
        canteen = sess2["location_canteen"] or "TBD"
        cancel_meetup_nag(sess.session_id)
        await safe_reply(msg, f"✅ Confirmed: {area}/{canteen} at {when}. Reminder {MEETUP_REMINDER_MINUTES} min before.")
        mark_replied(chat.id, user_id)
        return True

    return False

# ============================================================
# Main message handler: wake-only talk + ACTIVE listening
# ============================================================
async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    chat = update.effective_chat
    user = update.effective_user
    if not msg or not chat or not user or not msg.text:
        return

    text = msg.text.strip()
    text_l = text.lower()

    await ensure_chat_row(chat.id, chat.title or "")
    await touch_chat_activity(chat.id)
    await ensure_member_row(chat.id, user)
    await update_member_snippet(chat.id, user.id, text)

    bot_username = (context.bot.username or "")
    mentioned = is_bot_mentioned(msg, bot_username)
    replied_to_bot = bool(msg.reply_to_message and msg.reply_to_message.from_user and msg.reply_to_message.from_user.id == context.bot.id)
    wake_phrase = wake_phrase_present(text)

    dlog(
        "INCOMING_TEXT",
        chat_id=chat.id,
        chat_type=chat.type,
        chat_title=(chat.title or ""),
        user_id=user.id,
        username=(user.username or ""),
        name=(user.first_name or ""),
        mentioned=mentioned,
        replied=replied_to_bot,
        wake=wake_phrase,
        active=await is_active(chat.id),
        text=clip(text, 120),
    )

    # Store memory always (listens to everything)
    who = user.first_name or user.username or "someone"
    await remember(chat.id, "user", f"{who}: {clip(text)}")
    dlog("MEMORY_SAVED", chat_id=chat.id)

    # Mute keyword: if addressed, let them dismiss quickly
    if mentioned or replied_to_bot or wake_phrase:
        if any(k in text_l for k in DISMISS_KEYWORDS):
            await clear_active(chat.id)
            if not should_rate_limit(chat.id, user.id):
                await safe_reply(msg, "😴 Ok I’ll stop listening. Mention me if needed.")
                mark_replied(chat.id, user.id)
            return

    # If muted, never speak
    if await is_muted(chat.id):
        dlog("RETURN_MUTED", chat_id=chat.id)
        return

    # Wake conditions: mention OR reply-to-bot OR wake phrase.
    addressed = mentioned or replied_to_bot or wake_phrase

    # If addressed, enter/extend ACTIVE mode
    if addressed:
        await set_active(chat.id, ACTIVE_WINDOW_MIN)

    # While ACTIVE: listen for meetup changes when relevant
    if await is_active(chat.id):
        # If they say "stop" etc inside active mode but not necessarily addressed
        if any(k in text_l for k in MUTE_KEYWORDS) and not addressed:
            # do not auto-mute from passive chat; require address to avoid accidental triggers
            dlog("PASSIVE_MUTE_IGNORED", chat_id=chat.id)
        else:
            handled = await maybe_handle_meetup_from_chat(update, context, text)
            if handled:
                dlog("MEETUP_HANDLED", chat_id=chat.id)
                return

    # Outside ACTIVE, or if meetup logic didn't act:
    # TALK POLICY (your request #3): only respond when addressed.
    if not addressed:
        dlog("RETURN_NOT_ADDRESSED", chat_id=chat.id)
        return

    # Quiet hours: still allow direct addressed replies, but keep them short
    if should_rate_limit(chat.id, user.id):
        dlog("RETURN_RATE_LIMIT", chat_id=chat.id, user_id=user.id)
        return

    # If they say stop/mute/chill while addressed: mute bot
    if any(k in text_l for k in MUTE_KEYWORDS):
        await mute_chat(chat.id, DEFAULT_MUTE_MIN)
        await safe_reply(msg, f"ok ok… muted for {DEFAULT_MUTE_MIN} min 😇")
        mark_replied(chat.id, user.id)
        return

    # Normal addressed reply (banter/help)
    roast_level = 2  # keep safe default; you can add /roast_level back if you want
    reply = await llm_chat_reply(chat.id, roast_level, who, clip(text, 500))
    if not reply:
        reply = "👀 I’m here. Say “bot start meetup” or use /meetup."
    await safe_reply(msg, reply)
    await remember(chat.id, "assistant", reply)
    mark_replied(chat.id, user.id)
    dlog("REPLIED", chat_id=chat.id)

# ============================================================
# Error handler
# ============================================================
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Exception while handling an update:", exc_info=context.error)

# ============================================================
# Restore jobs on startup
# ============================================================
async def restore_jobs(application: Application):
    active = await db_fetchall(
        "SELECT session_id, chat_id, nag_interval_min FROM meetup_sessions WHERE status NOT IN ('confirmed','cancelled')"
    )
    for s in active:
        schedule_meetup_nag(application, int(s["chat_id"]), s["session_id"], int(s["nag_interval_min"] or MEETUP_DEFAULT_NAG_INTERVAL_MIN))
    logger.info("Restored %d meetup nag jobs", len(active))

    confirmed = await db_fetchall(
        "SELECT session_id, chat_id, chosen_option_id, reminder_sent FROM meetup_sessions WHERE status='confirmed' AND reminder_sent=0"
    )
    restored = 0
    for s in confirmed:
        session_id = s["session_id"]
        chat_id = int(s["chat_id"])
        chosen = s["chosen_option_id"]
        if not chosen:
            continue
        opt = await db_fetchone("SELECT start_ts FROM meetup_options WHERE option_id=? AND session_id=?", (chosen, session_id))
        if not opt:
            continue
        await schedule_meetup_reminder(application, chat_id, session_id, int(opt["start_ts"]))
        restored += 1
    logger.info("Restored %d meetup reminder jobs", restored)

# ============================================================
# Entry
# ============================================================
def main():
    init_db()

    request = HTTPXRequest(
        connect_timeout=20.0,
        read_timeout=30.0,
        write_timeout=30.0,
        pool_timeout=30.0,
    )

    app = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .request(request)
        .post_init(restore_jobs)
        .build()
    )

    # Commands
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("ping", ping))
    app.add_handler(CommandHandler("dismiss", dismiss))
    app.add_handler(CommandHandler("sleep", dismiss))
    app.add_handler(CommandHandler("meetup", meetup_cmd))
    app.add_handler(CommandHandler("meetup_status", meetup_status))
    app.add_handler(CommandHandler("meetup_cancel", meetup_cancel))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("openai_test", openai_test))

    # Callback query handler (meetup widgets)
    app.add_handler(CallbackQueryHandler(meetup_callback))

    # Message handler
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))

    # Error handler
    app.add_error_handler(error_handler)

    logger.info("Bot starting...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
