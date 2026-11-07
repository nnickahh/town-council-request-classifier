# Town Council Telegram Bot (RAG + ticketing)

## Setup
1. `python -m venv .venv && source .venv/bin/activate`
2. `pip install -r requirements.txt`
3. Create a bot with @BotFather, copy the token. Copy `.env.example` to `.env` and fill it in.
4. Put your council documents in `data/docs/` (format: see the sample files), then run `python ingest.py`
5. Create a Telegram group for staff, add the bot, send `/chatid` there, put the number in `.env` as `STAFF_CHAT_ID`.
6. `python bot.py`

## Try it
- "When is bulky waste collected?" -> answer with source and 👍/👎
- "There's a broken streetlight on Elm Street" -> ticket flow (location -> photo -> ticket #)
- "What's the best pizza place?" -> "I'm not sure", forwarded to staff
- Staff group: `/progress 1 crew booked`, `/resolve 1 fixed`, `/reply <user id> text`

## Files
- `bot.py` Telegram handlers, ticket flow, staff commands, rate limiting
- `rag.py` retrieval, grounded answering, intent classification
- `ingest.py` chunk + embed documents into Chroma
- `db.py` SQLite (tickets, logs, feedback)
- `eval.py` + `tests.json` accuracy and refusal test

## Tuning
Run `python eval.py` and look at the `dist=` column. Set `MAX_DISTANCE` in `.env` between the distances of
questions the bot should answer and ones it should refuse.

## Next steps
Streamlit staff dashboard, PDF ingestion (pypdf), multilingual emergency keywords, webhook deployment.

## Staff dashboard (Streamlit)
`streamlit run dashboard.py` opens a live page with tickets (photo, notes, In progress / Resolve buttons),
unanswered questions (type an answer, optionally add it to the knowledge base) and insights.
Status changes and answers are sent to residents through the bot automatically.
`python seed_demo.py` fills it with demo data. See `DEMO.md` for the hackathon script, pitch and judge Q&A.
If you add staff answers to the knowledge base while the bot runs and it doesn't pick them up, restart `python bot.py`.
