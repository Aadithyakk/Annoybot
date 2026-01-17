import os
import time
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
from telegram.error import TimedOut, NetworkError
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

# ============================================================
# Env vars
# ============================================================
TELEGRAM_BOT_TOKEN = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("Missing TELEGRAM_BOT_TOKEN. Put it in .env")

OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()
USE_OPENAI = bool(OPENAI_API_KEY)

DB_PATH = (os.getenv("BOT_DB_PATH") or "bot_state.sqlite3").strip()
BOT_TZ = os.getenv("BOT_TZ") or "Asia/Singapore"

# ============================================================
# Optional OpenAI (Responses API)
# ============================================================
oai_client = None
OPENAI_MODEL = os.getenv("OPENAI_MODEL") or "gpt-4.1-mini"

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
# Behavior knobs (tune these)
# ============================================================
CHAT_REPLY_COOLDOWN_SEC = int(os.getenv("CHAT_REPLY_COOLDOWN_SEC") or 20)
CHAT_MAX_REPLIES_PER_10MIN = int(os.getenv("CHAT_MAX_REPLIES_PER_10MIN") or 12)
USER_REPLY_COOLDOWN_SEC = int(os.getenv("USER_REPLY_COOLDOWN_SEC") or 60)

QUIET_HOUR_START = int(os.getenv("QUIET_HOUR_START") or 1)  # 1am
QUIET_HOUR_END = int(os.getenv("QUIET_HOUR_END") or 9)      # 9am

MAX_MEMORY_MESSAGES = int(os.getenv("MAX_MEMORY_MESSAGES") or 20)
PERSIST_MEMORY_MESSAGES = int(os.getenv("PERSIST_MEMORY_MESSAGES") or 60)  # stored per chat

DEFAULT_MUTE_MIN = int(os.getenv("DEFAULT_MUTE_MIN") or 30)

MUTE_KEYWORDS = ("stop", "stfu", "shut up", "mute bot", "no bot", "quiet bot", "chill")
WAKE_WORDS = ("bot", "annoyotron", "oi bot")
TRIGGER_WORDS = ("where eat", "lunch", "dinner", "meetup", "meet", "hungry", "canteen")

# AutoPoke: bot speaks on its own
AUTOPOKE_DEFAULT_INTERVAL_MIN = int(os.getenv("AUTOPOKE_DEFAULT_INTERVAL_MIN") or 180)
AUTOPOKE_MIN_SILENCE_MIN = int(os.getenv("AUTOPOKE_MIN_SILENCE_MIN") or 30)

# Meetup
MEETUP_DEFAULT_NAG_INTERVAL_MIN = int(os.getenv("MEETUP_DEFAULT_NAG_INTERVAL_MIN") or 5)
MEETUP_MAX_MENTIONS = int(os.getenv("MEETUP_MAX_MENTIONS") or 5)

# Suggest time slots (simple + practical)
MEETUP_TIME_SLOTS = [
    "12:00", "12:30", "13:00",
    "18:00", "18:30", "19:00",
]

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
        admin_only INTEGER DEFAULT 0
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

    # Persistent memory (last messages for LLM context)
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

    # Meetup tables
    c.execute("""
    CREATE TABLE IF NOT EXISTS meetup_sessions (
        session_id TEXT PRIMARY KEY,
        chat_id INTEGER,
        status TEXT, -- 'collecting_area', 'collecting_canteen', 'collecting_time', 'pending_confirm', 'confirmed', 'cancelled'
        area TEXT,
        canteen TEXT,
        time TEXT,
        created_by INTEGER,
        created_ts INTEGER,
        confirmed_ts INTEGER,
        nag_interval_min INTEGER DEFAULT 5
    )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_meetup_chat_status ON meetup_sessions(chat_id, status)")

    c.execute("""
    CREATE TABLE IF NOT EXISTS meetup_votes (
        session_id TEXT,
        user_id INTEGER,
        vote_type TEXT, -- 'area', 'canteen', 'time'
        vote_value TEXT,
        PRIMARY KEY (session_id, user_id, vote_type)
    )
    """)

    # Migrations for older DBs (idempotent)
    _add_column_if_missing(conn, "chats", "muted_until_ts INTEGER DEFAULT 0")
    _add_column_if_missing(conn, "chats", "admin_only INTEGER DEFAULT 0")

    _add_column_if_missing(conn, "members", "last_message_ts INTEGER DEFAULT 0")
    _add_column_if_missing(conn, "members", "last_snippet TEXT DEFAULT ''")

    _add_column_if_missing(conn, "meetup_sessions", "canteen TEXT")
    _add_column_if_missing(conn, "meetup_sessions", "nag_interval_min INTEGER DEFAULT 5")

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
# In-memory rate limit
# ============================================================
chat_last_reply_ts: Dict[int, int] = {}
chat_reply_times: Dict[int, List[int]] = {}
user_last_reply_ts: Dict[Tuple[int, int], int] = {}

def now_ts() -> int:
    return int(time.time())

def clip(text: str, n: int = 180) -> str:
    t = (text or "").replace("\n", " ").strip()
    return t[:n] + ("…" if len(t) > n else "")

def local_hour() -> int:
    tz = ZoneInfo(BOT_TZ)
    return datetime.now(tz).hour

def in_quiet_hours() -> bool:
    h = local_hour()
    if QUIET_HOUR_START < QUIET_HOUR_END:
        return QUIET_HOUR_START <= h < QUIET_HOUR_END
    return h >= QUIET_HOUR_START or h < QUIET_HOUR_END

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
    # prune
    while times and times[0] < cutoff:
        times.pop(0)
    if len(times) >= CHAT_MAX_REPLIES_PER_10MIN:
        return True

    return False

def mark_replied(chat_id: int, user_id: int):
    now = now_ts()
    chat_last_reply_ts[chat_id] = now
    user_last_reply_ts[(chat_id, user_id)] = now
    times = chat_reply_times.setdefault(chat_id, [])
    times.append(now)

# ============================================================
# Persistent memory for LLM context
# ============================================================
async def remember(chat_id: int, role: str, content: str):
    # store clipped for safety
    content = clip(content, 500)
    await db_exec(
        "INSERT INTO messages(chat_id, ts, role, content) VALUES (?, ?, ?, ?)",
        (chat_id, now_ts(), role, content),
    )
    # keep table bounded per chat
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
    # Only last MAX_MEMORY_MESSAGES for prompt
    trimmed = rows[-MAX_MEMORY_MESSAGES:]
    return [{"role": r["role"], "content": r["content"]} for r in trimmed]

# ============================================================
# Telegram send safety (prevents crash on timeouts)
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

# ============================================================
# OpenAI: generate roasty-but-safe group-member replies
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

Goals:
- Reply like a normal group member, not a helpdesk.
- Move the convo forward with banter and a bit of chaos.
- Occasionally remind about /help, /autopoke_on, /chaos_on.
"""

async def llm_reply(chat_id: int, roast_level: int, user_name: str, text: str) -> Optional[str]:
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
        logger.warning("LLM call failed: %s", e)
        return None

# ============================================================
# Chat + member helpers
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
    await safe_reply(update.effective_message, "⛔ Admin-only setting is ON. Ask an admin to do that.")
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
# Commands
# ============================================================
HELP_TEXT = (
    "Annoyotron commands:\n"
    "/chaos_on – I start chatting like a real group member\n"
    "/chaos_off – I shut up (mostly)\n"
    "/roast_level <1-5> – set roast intensity\n"
    "/roast_optin – you consent to slightly spicier teasing\n"
    "/roast_optout – opt out\n"
    "/autopoke_on [minutes] – I will message on my own schedule\n"
    "/autopoke_off – stop autopoke\n"
    "/mute [minutes] – mute me for the chat\n"
    "/unmute – unmute me\n"
    "/admin_only_on – only admins can change bot settings\n"
    "/admin_only_off – disable admin-only gate\n"
    "/meetup [nag_interval_min] – start lunch/dinner planning\n"
    "/meetup_status – show current meetup\n"
    "/meetup_cancel – cancel current meetup\n"
    "/openai_test – test OpenAI from Telegram\n"
    "/status – show current settings\n"
)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await safe_reply(
        update.effective_message,
        "✅ Annoyotron online.\n"
        "For normal group-chat replies, disable Bot Privacy:\n"
        "BotFather → /setprivacy → Disable\n\n"
        + HELP_TEXT
    )

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await safe_reply(update.effective_message, HELP_TEXT)

async def chaos_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await admin_gate(update, context):
        return
    chat = update.effective_chat
    await ensure_chat_row(chat.id, chat.title or "")
    await db_exec("UPDATE chats SET chaos_enabled=1 WHERE chat_id=?", (chat.id,))
    await safe_reply(update.effective_message, "😈 Chaos mode ON. I will talk like a group member (rate-limited).")

async def chaos_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await admin_gate(update, context):
        return
    chat = update.effective_chat
    await ensure_chat_row(chat.id, chat.title or "")
    await db_exec("UPDATE chats SET chaos_enabled=0 WHERE chat_id=?", (chat.id,))
    await safe_reply(update.effective_message, "😇 Chaos mode OFF. I’ll mostly only respond to /commands.")

async def roast_level(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await admin_gate(update, context):
        return
    chat = update.effective_chat
    await ensure_chat_row(chat.id, chat.title or "")
    if not context.args:
        await safe_reply(update.effective_message, "Usage: /roast_level 1-5")
        return
    try:
        lvl = int(context.args[0])
        lvl = max(1, min(5, lvl))
    except ValueError:
        await safe_reply(update.effective_message, "Give a number 1-5.")
        return
    await db_exec("UPDATE chats SET roast_level=? WHERE chat_id=?", (lvl, chat.id))
    await safe_reply(update.effective_message, f"🔥 Roast level set to {lvl}/5.")

async def roast_optin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat
    user = update.effective_user
    await ensure_chat_row(chat.id, chat.title or "")
    await ensure_member_row(chat.id, user)
    await db_exec("UPDATE members SET roast_opt_in=1 WHERE chat_id=? AND user_id=?", (chat.id, user.id))
    await safe_reply(update.effective_message, "✅ Opted into spicier teasing.")

async def roast_optout(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat
    user = update.effective_user
    await ensure_chat_row(chat.id, chat.title or "")
    await ensure_member_row(chat.id, user)
    await db_exec("UPDATE members SET roast_opt_in=0 WHERE chat_id=? AND user_id=?", (chat.id, user.id))
    await safe_reply(update.effective_message, "✅ Opted out. I’ll keep it calmer with you.")

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
        tz = ZoneInfo(BOT_TZ)
        muted_str = datetime.fromtimestamp(muted_until, tz).strftime("%Y-%m-%d %H:%M")

    await safe_reply(
        update.effective_message,
        f"Settings:\n"
        f"- chaos_enabled: {int(row['chaos_enabled'])}\n"
        f"- roast_level: {int(row['roast_level'])}/5\n"
        f"- openai_enabled: {USE_OPENAI}\n"
        f"- autopoke_enabled: {int(row['autopoke_enabled'])}\n"
        f"- autopoke_interval_min: {int(row['autopoke_interval_min'])}\n"
        f"- quiet_hours: {QUIET_HOUR_START}:00–{QUIET_HOUR_END}:00 ({BOT_TZ})\n"
        f"- muted_until: {muted_str}\n"
        f"- admin_only: {int(row['admin_only'])}\n"
        f"- cooldown: chat {CHAT_REPLY_COOLDOWN_SEC}s, user {USER_REPLY_COOLDOWN_SEC}s\n"
        f"- cap: {CHAT_MAX_REPLIES_PER_10MIN} replies/10min"
    )

async def openai_test(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not USE_OPENAI or not oai_client:
        await safe_reply(
            update.effective_message,
            "OpenAI not available. Check:\n"
            "- pip install -U openai\n"
            "- OPENAI_API_KEY in .env\n"
        )
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

async def mute_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await admin_gate(update, context):
        return
    chat = update.effective_chat
    await ensure_chat_row(chat.id, chat.title or "")
    minutes = DEFAULT_MUTE_MIN
    if context.args:
        try:
            minutes = int(context.args[0])
            minutes = max(1, min(24 * 60, minutes))
        except ValueError:
            await safe_reply(update.effective_message, "Usage: /mute [minutes] (e.g. /mute 30)")
            return
    await mute_chat(chat.id, minutes)
    await safe_reply(update.effective_message, f"🤐 Ok lah. Muted for {minutes} min.")

async def unmute_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await admin_gate(update, context):
        return
    chat = update.effective_chat
    await ensure_chat_row(chat.id, chat.title or "")
    await unmute_chat(chat.id)
    await safe_reply(update.effective_message, "🔊 Unmuted. I’m back 😈")

async def admin_only_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_admin(update, context):
        await safe_reply(update.effective_message, "⛔ Only chat admins can enable admin-only mode.")
        return
    chat = update.effective_chat
    await ensure_chat_row(chat.id, chat.title or "")
    await db_exec("UPDATE chats SET admin_only=1 WHERE chat_id=?", (chat.id,))
    await safe_reply(update.effective_message, "🛡️ Admin-only mode ON. Only admins can change bot settings.")

async def admin_only_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_admin(update, context):
        await safe_reply(update.effective_message, "⛔ Only chat admins can disable admin-only mode.")
        return
    chat = update.effective_chat
    await ensure_chat_row(chat.id, chat.title or "")
    await db_exec("UPDATE chats SET admin_only=0 WHERE chat_id=?", (chat.id,))
    await safe_reply(update.effective_message, "🛡️ Admin-only mode OFF. Anyone can change bot settings again.")

# ============================================================
# AutoPoke scheduling (JobQueue)
# ============================================================
_autopoke_jobs: Dict[int, object] = {}

async def autopoke_job(context: ContextTypes.DEFAULT_TYPE):
    chat_id = context.job.chat_id

    if in_quiet_hours():
        return

    row = await db_fetchone("SELECT * FROM chats WHERE chat_id=?", (chat_id,))
    if not row:
        return
    if int(row["autopoke_enabled"] or 0) != 1:
        return
    if int(row["chaos_enabled"] or 0) != 1:
        # if chaos off, don't autopoke
        return
    if now_ts() < int(row["muted_until_ts"] or 0):
        return

    last_act = int(row["last_activity_ts"] or 0)
    if last_act > 0:
        silence_min = (now_ts() - last_act) // 60
        if silence_min < AUTOPOKE_MIN_SILENCE_MIN:
            return

    roast_lvl = int(row["roast_level"] or 2)

    # Rate-limit as "user_id=0"
    if should_rate_limit(chat_id, 0):
        return

    prompt = "The group chat has been quiet. Say ONE short message to revive the chat. Do not target individuals."
    reply = await llm_reply(chat_id, roast_lvl, "system", prompt)

    if not reply:
        reply = random.choice([
            "Wah this chat went offline ah 😭 anyone alive?",
            "Bro… everyone sleeping isit. Talk cock a bit leh.",
            "Hello? I can hear the silence sia.",
        ])

    await safe_send(context.bot, chat_id, reply)
    await remember(chat_id, "assistant", reply)
    mark_replied(chat_id, 0)

def _schedule_autopoke(application: Application, chat_id: int, interval_min: int):
    old = _autopoke_jobs.get(chat_id)
    if old is not None:
        try:
            old.schedule_removal()
        except Exception:
            pass
        _autopoke_jobs.pop(chat_id, None)

    job = application.job_queue.run_repeating(
        autopoke_job,
        interval=max(5, interval_min) * 60,
        first=30,
        chat_id=chat_id,
        name=f"autopoke_{chat_id}",
    )
    _autopoke_jobs[chat_id] = job

async def autopoke_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await admin_gate(update, context):
        return
    chat = update.effective_chat
    await ensure_chat_row(chat.id, chat.title or "")

    interval_min = AUTOPOKE_DEFAULT_INTERVAL_MIN
    if context.args:
        try:
            interval_min = int(context.args[0])
            interval_min = max(5, min(24 * 60, interval_min))
        except ValueError:
            await safe_reply(update.effective_message, "Usage: /autopoke_on [minutes] (e.g. /autopoke_on 180)")
            return

    await db_exec(
        "UPDATE chats SET autopoke_enabled=1, autopoke_interval_min=? WHERE chat_id=?",
        (interval_min, chat.id)
    )

    _schedule_autopoke(context.application, chat.id, interval_min)
    await safe_reply(update.effective_message, f"🕒 AutoPoke ON. I’ll poke every ~{interval_min} min (only if chat is quiet).")

async def autopoke_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await admin_gate(update, context):
        return
    chat = update.effective_chat
    await ensure_chat_row(chat.id, chat.title or "")
    await db_exec("UPDATE chats SET autopoke_enabled=0 WHERE chat_id=?", (chat.id,))

    old = _autopoke_jobs.get(chat.id)
    if old is not None:
        try:
            old.schedule_removal()
        except Exception:
            pass
        _autopoke_jobs.pop(chat.id, None)

    await safe_reply(update.effective_message, "🛑 AutoPoke OFF.")

async def restore_jobs(application: Application):
    # Restore autopoke jobs
    rows = await db_fetchall(
        "SELECT chat_id, autopoke_enabled, autopoke_interval_min FROM chats WHERE autopoke_enabled=1"
    )
    for r in rows:
        chat_id = int(r["chat_id"])
        interval_min = int(r["autopoke_interval_min"] or AUTOPOKE_DEFAULT_INTERVAL_MIN)
        _schedule_autopoke(application, chat_id, interval_min)
    logger.info("Restored %d autopoke jobs", len(rows))

    # Restore any active meetup nag jobs
    active = await db_fetchall(
        "SELECT session_id, chat_id, nag_interval_min FROM meetup_sessions WHERE status NOT IN ('confirmed','cancelled')"
    )
    for s in active:
        session_id = s["session_id"]
        chat_id = int(s["chat_id"])
        nag = int(s["nag_interval_min"] or MEETUP_DEFAULT_NAG_INTERVAL_MIN)
        _schedule_meetup_nag(application, chat_id, session_id, nag)
    if active:
        logger.info("Restored %d meetup nag jobs", len(active))

# ============================================================
# Meetup flow (Area -> Canteen -> Time -> Confirm/Cancel)
# Callback data format:
#   "m|<sid>|<action>|<value>"
# ============================================================
_meetup_nag_jobs: Dict[str, object] = {}  # key: session_id

def _mk_cb(session_id: str, action: str, value: str = "") -> str:
    # Keep callback short (Telegram limit ~64 bytes)
    if value:
        return f"m|{session_id}|{action}|{value}"
    return f"m|{session_id}|{action}|"

async def _get_active_session(chat_id: int):
    return await db_fetchone(
        "SELECT * FROM meetup_sessions WHERE chat_id=? AND status NOT IN ('confirmed','cancelled') ORDER BY created_ts DESC LIMIT 1",
        (chat_id,),
    )

def _schedule_meetup_nag(application: Application, chat_id: int, session_id: str, nag_interval_min: int):
    key = session_id
    old = _meetup_nag_jobs.get(key)
    if old is not None:
        try:
            old.schedule_removal()
        except Exception:
            pass
        _meetup_nag_jobs.pop(key, None)

    job = application.job_queue.run_repeating(
        meetup_nag_job,
        interval=max(1, nag_interval_min) * 60,
        first=max(30, nag_interval_min * 30),
        chat_id=chat_id,
        name=f"meetup_nag_{chat_id}_{session_id}",
        data={"session_id": session_id},
    )
    _meetup_nag_jobs[key] = job

def _cancel_meetup_nag(session_id: str):
    job = _meetup_nag_jobs.get(session_id)
    if job is not None:
        try:
            job.schedule_removal()
        except Exception:
            pass
        _meetup_nag_jobs.pop(session_id, None)

async def meetup_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat
    user = update.effective_user
    await ensure_chat_row(chat.id, chat.title or "")

    if chat.type not in (ChatType.GROUP, ChatType.SUPERGROUP):
        await safe_reply(update.effective_message, "Meetup works best in a group chat lah.")
        return

    # Block if active exists
    existing = await _get_active_session(chat.id)
    if existing:
        await safe_reply(update.effective_message, "🚨 A meetup is already being planned! Use /meetup_status or /meetup_cancel.")
        return

    nag_interval = MEETUP_DEFAULT_NAG_INTERVAL_MIN
    if context.args:
        try:
            nag_interval = int(context.args[0])
            nag_interval = max(1, min(60, nag_interval))
        except ValueError:
            await safe_reply(update.effective_message, "Usage: /meetup [nag_interval_minutes] (e.g. /meetup 3)")
            return

    # short session id
    session_id = secrets.token_hex(4)  # 8 hex chars
    await db_exec(
        """
        INSERT INTO meetup_sessions(session_id, chat_id, status, created_by, created_ts, nag_interval_min)
        VALUES (?, ?, 'collecting_area', ?, ?, ?)
        """,
        (session_id, chat.id, user.id, now_ts(), nag_interval),
    )

    # Area buttons
    keyboard = []
    row = []
    for area in CANTEENS.keys():
        row.append(InlineKeyboardButton(area, callback_data=_mk_cb(session_id, "area", area)))
        if len(row) == 2:
            keyboard.append(row)
            row = []
    if row:
        keyboard.append(row)

    await safe_send(
        context.bot,
        chat.id,
        f"📍 **MEETUP TIME.** Where we going? Vote area first.\n(Nagging every {nag_interval} min until confirmed)",
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode="Markdown",
    )

    _schedule_meetup_nag(context.application, chat.id, session_id, nag_interval)

async def meetup_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat
    if chat.type not in (ChatType.GROUP, ChatType.SUPERGROUP):
        await safe_reply(update.effective_message, "No meetup status in private chat leh.")
        return

    sess = await _get_active_session(chat.id)
    if not sess:
        await safe_reply(update.effective_message, "No active meetup. Use /meetup to start one.")
        return

    area = sess["area"] or "-"
    canteen = sess["canteen"] or "-"
    t = sess["time"] or "-"
    status = sess["status"]
    await safe_reply(update.effective_message, f"📍 Meetup status: **{status}**\nArea: {area}\nCanteen: {canteen}\nTime: {t}", parse_mode="Markdown")

async def meetup_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat
    if chat.type not in (ChatType.GROUP, ChatType.SUPERGROUP):
        return
    sess = await _get_active_session(chat.id)
    if not sess:
        await safe_reply(update.effective_message, "No active meetup to cancel.")
        return

    session_id = sess["session_id"]
    await db_exec("UPDATE meetup_sessions SET status='cancelled' WHERE session_id=?", (session_id,))
    _cancel_meetup_nag(session_id)
    await safe_reply(update.effective_message, "🛑 Meetup cancelled. Y’all too chaotic sia.")

async def meetup_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    data = query.data or ""
    # Expected: m|sid|action|value
    parts = data.split("|", 3)
    if len(parts) < 3 or parts[0] != "m":
        return

    session_id = parts[1]
    action = parts[2]
    value = parts[3] if len(parts) > 3 else ""
    user = query.from_user
    chat_id = query.message.chat_id if query.message else None
    if not chat_id:
        return

    sess = await db_fetchone("SELECT * FROM meetup_sessions WHERE session_id=?", (session_id,))
    if not sess or int(sess["chat_id"]) != int(chat_id):
        await query.edit_message_text("This meetup is no longer active.")
        return
    if sess["status"] in ("confirmed", "cancelled"):
        await query.edit_message_text("This meetup is already closed.")
        return

    # Record vote (even if we also set session fields)
    vote_type = action  # area/canteen/time
    if vote_type not in ("area", "canteen", "time", "confirm", "cancel"):
        return

    if action == "cancel":
        await db_exec("UPDATE meetup_sessions SET status='cancelled' WHERE session_id=?", (session_id,))
        _cancel_meetup_nag(session_id)
        await query.edit_message_text("🛑 Meetup cancelled. Next time earlier decide lah.")
        return

    if action == "confirm":
        # only allow confirm if pending_confirm
        if sess["status"] != "pending_confirm":
            await query.answer("Not ready to confirm yet.", show_alert=False)
            return
        await db_exec(
            "UPDATE meetup_sessions SET status='confirmed', confirmed_ts=? WHERE session_id=?",
            (now_ts(), session_id),
        )
        _cancel_meetup_nag(session_id)
        await query.edit_message_text(
            f"✅ **MEETUP CONFIRMED**\n📍 {sess['area']} — {sess['canteen']}\n🕒 {sess['time']}\n\nOk settle. See you snakes there 🐍",
            parse_mode="Markdown",
        )
        return

    # Voting steps
    if action == "area":
        # set area, move to collecting_canteen
        if value not in CANTEENS:
            await query.answer("Invalid area", show_alert=False)
            return

        await db_exec("""
            INSERT INTO meetup_votes(session_id, user_id, vote_type, vote_value)
            VALUES(?, ?, 'area', ?)
            ON CONFLICT(session_id, user_id, vote_type) DO UPDATE SET vote_value=excluded.vote_value
        """, (session_id, user.id, value))

        await db_exec("UPDATE meetup_sessions SET area=?, status='collecting_canteen' WHERE session_id=?", (value, session_id))

        # show canteen options
        cans = CANTEENS[value]
        kb = []
        r = []
        for c in cans:
            r.append(InlineKeyboardButton(c, callback_data=_mk_cb(session_id, "canteen", c)))
            if len(r) == 2:
                kb.append(r); r = []
        if r: kb.append(r)
        kb.append([InlineKeyboardButton("Cancel meetup", callback_data=_mk_cb(session_id, "cancel", ""))])

        await query.edit_message_text(
            f"📍 Area locked-ish: **{value}**\nNow pick canteen:",
            reply_markup=InlineKeyboardMarkup(kb),
            parse_mode="Markdown",
        )
        return

    if action == "canteen":
        area = sess["area"]
        if not area or area not in CANTEENS:
            await query.answer("Pick area first", show_alert=False)
            return
        if value not in CANTEENS[area]:
            await query.answer("Invalid canteen", show_alert=False)
            return

        await db_exec("""
            INSERT INTO meetup_votes(session_id, user_id, vote_type, vote_value)
            VALUES(?, ?, 'canteen', ?)
            ON CONFLICT(session_id, user_id, vote_type) DO UPDATE SET vote_value=excluded.vote_value
        """, (session_id, user.id, value))

        await db_exec("UPDATE meetup_sessions SET canteen=?, status='collecting_time' WHERE session_id=?", (value, session_id))

        # time slots buttons
        kb = []
        r = []
        for t in MEETUP_TIME_SLOTS:
            r.append(InlineKeyboardButton(t, callback_data=_mk_cb(session_id, "time", t)))
            if len(r) == 3:
                kb.append(r); r = []
        if r: kb.append(r)
        kb.append([InlineKeyboardButton("Cancel meetup", callback_data=_mk_cb(session_id, "cancel", ""))])

        await query.edit_message_text(
            f"📍 **{area} — {value}**\nNow pick time:",
            reply_markup=InlineKeyboardMarkup(kb),
            parse_mode="Markdown",
        )
        return

    if action == "time":
        # set time, go to pending_confirm
        await db_exec("""
            INSERT INTO meetup_votes(session_id, user_id, vote_type, vote_value)
            VALUES(?, ?, 'time', ?)
            ON CONFLICT(session_id, user_id, vote_type) DO UPDATE SET vote_value=excluded.vote_value
        """, (session_id, user.id, value))

        await db_exec("UPDATE meetup_sessions SET time=?, status='pending_confirm' WHERE session_id=?", (value, session_id))

        # Refresh sess for display
        sess2 = await db_fetchone("SELECT * FROM meetup_sessions WHERE session_id=?", (session_id,))
        kb = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("✅ Confirm", callback_data=_mk_cb(session_id, "confirm", "")),
                InlineKeyboardButton("❌ Cancel", callback_data=_mk_cb(session_id, "cancel", "")),
            ]
        ])

        await query.edit_message_text(
            f"**Final check:**\n📍 {sess2['area']} — {sess2['canteen']}\n🕒 {sess2['time']}\n\nConfirm or cancel:",
            reply_markup=kb,
            parse_mode="Markdown",
        )
        return

async def meetup_nag_job(context: ContextTypes.DEFAULT_TYPE):
    chat_id = context.job.chat_id
    data = context.job.data or {}
    session_id = data.get("session_id")

    if in_quiet_hours():
        return
    if not session_id:
        return

    sess = await db_fetchone("SELECT * FROM meetup_sessions WHERE session_id=?", (session_id,))
    if not sess or sess["status"] in ("confirmed", "cancelled"):
        _cancel_meetup_nag(session_id)
        return

    # Get members in the chat (only those we've seen)
    all_members = await db_fetchall("SELECT user_id, username, first_name FROM members WHERE chat_id=?", (chat_id,))
    if not all_members:
        return

    # Determine which vote_type is currently relevant
    status = sess["status"]
    if status == "collecting_area":
        needed = "area"
    elif status == "collecting_canteen":
        needed = "canteen"
    elif status == "collecting_time":
        needed = "time"
    elif status == "pending_confirm":
        # no nagging for confirm; too spammy
        return
    else:
        return

    voted_users = await db_fetchall(
        "SELECT DISTINCT user_id FROM meetup_votes WHERE session_id=? AND vote_type=?",
        (session_id, needed),
    )
    voted_ids = {row["user_id"] for row in voted_users}
    non_voters = [m for m in all_members if m["user_id"] not in voted_ids]

    if not non_voters:
        return

    mentions = []
    for m in non_voters[:MEETUP_MAX_MENTIONS]:
        username = m["username"]
        if username:
            mentions.append(f"@{username}")
        else:
            mentions.append(m["first_name"] or "someone")

    stage_text = {
        "area": "pick area",
        "canteen": "pick canteen",
        "time": "pick time",
    }.get(needed, "vote")

    nag_text = f"Oi {', '.join(mentions)}! {stage_text} leh. Meetup planning not gonna finish by itself sia 🐍"
    await safe_send(context.bot, chat_id, nag_text)

# ============================================================
# Main message handler: LLM-style replies
# ============================================================
async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    chat = update.effective_chat
    user = update.effective_user
    if not msg or not chat or not user or not msg.text:
        return

    text = msg.text.strip()
    text_l = text.lower()

    # Track activity for groups
    if chat.type in (ChatType.GROUP, ChatType.SUPERGROUP):
        await ensure_chat_row(chat.id, chat.title or "")
        await touch_chat_activity(chat.id)
        await ensure_member_row(chat.id, user)
        await update_member_snippet(chat.id, user.id, text)

    # Persistent memory
    if chat.type in (ChatType.GROUP, ChatType.SUPERGROUP):
        who = user.first_name or user.username or "someone"
        await remember(chat.id, "user", f"{who}: {clip(text)}")
    else:
        await remember(chat.id, "user", clip(text))

    # Private chat: always respond (unless rate limited)
    if chat.type == ChatType.PRIVATE:
        if should_rate_limit(chat.id, user.id):
            return
        if any(k in text_l for k in MUTE_KEYWORDS):
            await safe_reply(msg, "ok ok, I’ll chill 😇")
            mark_replied(chat.id, user.id)
            return

        reply = await llm_reply(chat.id, 2, user.first_name or "User", clip(text, 500))
        if not reply:
            reply = "👀 I’m here. Try /help."
        await safe_reply(msg, reply)
        await remember(chat.id, "assistant", reply)
        mark_replied(chat.id, user.id)
        return

    # Group only beyond this point
    if chat.type not in (ChatType.GROUP, ChatType.SUPERGROUP):
        return

    settings = await db_fetchone("SELECT * FROM chats WHERE chat_id=?", (chat.id,))
    chaos_enabled = int(settings["chaos_enabled"] or 0) if settings else 0
    roast_lvl = int(settings["roast_level"] or 2) if settings else 2
    muted_until = int(settings["muted_until_ts"] or 0) if settings else 0

    if not chaos_enabled:
        return

    if now_ts() < muted_until:
        return

    if in_quiet_hours():
        return

    # If someone says mute keywords, actually mute for DEFAULT_MUTE_MIN
    if any(k in text_l for k in MUTE_KEYWORDS):
        if should_rate_limit(chat.id, user.id):
            return
        await mute_chat(chat.id, DEFAULT_MUTE_MIN)
        await safe_reply(msg, f"ok ok ok… I’ll shut up for {DEFAULT_MUTE_MIN} min 😇")
        mark_replied(chat.id, user.id)
        return

    # Decide when to respond
    bot_username = (context.bot.username or "").lower()
    mentioned = bot_username and (f"@{bot_username}" in text_l)

    replied_to_bot = False
    if msg.reply_to_message and msg.reply_to_message.from_user:
        replied_to_bot = (msg.reply_to_message.from_user.id == context.bot.id)

    wake = any(text_l.startswith(w + " ") for w in WAKE_WORDS)

    # Smarter interject: if trigger words OR chat is quiet-ish
    last_act = int(settings["last_activity_ts"] or 0) if settings else 0
    silence_min = (now_ts() - last_act) // 60 if last_act else 999

    trigger_hit = any(t in text_l for t in TRIGGER_WORDS)
    quietish = silence_min >= 10  # 10 mins quiet
    # low chance interject, higher if trigger/quiet
    base_p = 0.06
    if trigger_hit:
        base_p = 0.20
    elif quietish:
        base_p = 0.10

    interject = (random.random() < base_p)

    if not (mentioned or replied_to_bot or wake or interject):
        return

    if should_rate_limit(chat.id, user.id):
        return

    # Respect per-user opt-in/out
    mrow = await db_fetchone("SELECT roast_opt_in FROM members WHERE chat_id=? AND user_id=?", (chat.id, user.id))
    user_opt_in = int(mrow["roast_opt_in"] or 0) if mrow else 0
    effective_roast = roast_lvl if user_opt_in else max(1, roast_lvl - 1)

    reply = await llm_reply(
        chat.id,
        effective_roast,
        user.first_name or user.username or "someone",
        clip(text, 500),
        )

    if not reply:
        reply = random.choice([
            "👀 I’m here. Say my name or use /help.",
            "Eh what you want from me sia 😭",
            "Wah ok noted… but also skill issue lah.",
        ])

    await safe_reply(msg, reply)
    await remember(chat.id, "assistant", reply)
    mark_replied(chat.id, user.id)

# ============================================================
# Error handler
# ============================================================
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Exception while handling an update:", exc_info=context.error)

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
    app.add_handler(CommandHandler("chaos_on", chaos_on))
    app.add_handler(CommandHandler("chaos_off", chaos_off))
    app.add_handler(CommandHandler("roast_level", roast_level))
    app.add_handler(CommandHandler("roast_optin", roast_optin))
    app.add_handler(CommandHandler("roast_optout", roast_optout))
    app.add_handler(CommandHandler("autopoke_on", autopoke_on))
    app.add_handler(CommandHandler("autopoke_off", autopoke_off))
    app.add_handler(CommandHandler("mute", mute_cmd))
    app.add_handler(CommandHandler("unmute", unmute_cmd))
    app.add_handler(CommandHandler("admin_only_on", admin_only_on))
    app.add_handler(CommandHandler("admin_only_off", admin_only_off))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("openai_test", openai_test))

    # Meetup commands
    app.add_handler(CommandHandler("meetup", meetup_cmd))
    app.add_handler(CommandHandler("meetup_status", meetup_status))
    app.add_handler(CommandHandler("meetup_cancel", meetup_cancel))

    # Callback query handler for meetup buttons
    app.add_handler(CallbackQueryHandler(meetup_callback))

    # Message handler
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))

    # Error handler
    app.add_error_handler(error_handler)

    logger.info("Bot starting...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
